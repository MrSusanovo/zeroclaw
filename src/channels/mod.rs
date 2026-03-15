//! Channel subsystem for messaging platform integrations.
//!
//! This module provides the multi-channel messaging infrastructure that connects
//! ZeroClaw to external platforms. Each channel implements the [`Channel`] trait
//! defined in [`traits`], which provides a uniform interface for sending messages,
//! listening for incoming messages, health checking, and typing indicators.
//!
//! Channels are instantiated by [`start_channels`] based on the runtime configuration.
//! The subsystem manages per-sender conversation history, concurrent message processing
//! with configurable parallelism, and exponential-backoff reconnection for resilience.
//!
//! # Extension
//!
//! To add a new channel, implement [`Channel`] and [`ChannelFactory`] in a new submodule,
//! then register it in [`channel_factories`]. See `AGENTS.md` §7.2 for the full change playbook.

pub mod clawdtalk;
pub mod cli;
pub mod dingtalk;
pub mod discord;
pub mod email_channel;
pub mod imessage;
pub mod irc;
#[cfg(feature = "channel-lark")]
pub mod lark;
pub mod linq;
#[cfg(feature = "channel-matrix")]
pub mod matrix;
pub mod mattermost;
pub mod message_handler;
pub mod nextcloud_talk;
#[cfg(feature = "channel-nostr")]
pub mod nostr;
pub mod qq;
pub mod signal;
pub mod slack;
pub mod telegram;
pub mod traits;
pub mod transcription;
pub mod tts;
pub mod wati;
pub mod whatsapp;
#[cfg(feature = "whatsapp-web")]
pub mod whatsapp_storage;
#[cfg(feature = "whatsapp-web")]
pub mod whatsapp_web;

pub use clawdtalk::{ClawdTalkChannel, ClawdTalkConfig};
pub use cli::CliChannel;
pub use dingtalk::DingTalkChannel;
pub use discord::DiscordChannel;
pub use email_channel::EmailChannel;
pub use imessage::IMessageChannel;
pub use irc::IrcChannel;
#[cfg(feature = "channel-lark")]
pub use lark::LarkChannel;
pub use linq::LinqChannel;
#[cfg(feature = "channel-matrix")]
pub use matrix::MatrixChannel;
pub use mattermost::MattermostChannel;
pub use nextcloud_talk::NextcloudTalkChannel;
#[cfg(feature = "channel-nostr")]
pub use nostr::NostrChannel;
pub use qq::QQChannel;
pub use signal::SignalChannel;
pub use slack::SlackChannel;
pub use telegram::TelegramChannel;
pub use traits::{Channel, SendMessage};
#[allow(unused_imports)]
pub use tts::{TtsManager, TtsProvider};
pub use wati::WatiChannel;
pub use whatsapp::WhatsAppChannel;
#[cfg(feature = "whatsapp-web")]
pub use whatsapp_web::WhatsAppWebChannel;

use crate::agent::loop_::{build_tool_instructions, run_tool_call_loop};
use crate::config::Config;
use crate::identity;
use crate::memory::{self, Memory};
use crate::observability::traits::{ObserverEvent, ObserverMetric};
use crate::observability::{self, runtime_trace, Observer};
use crate::providers::{self, ChatMessage, Provider};
use crate::runtime;
use crate::security::SecurityPolicy;
use crate::tools::{self, Tool};
use crate::util::truncate_with_ellipsis;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use tokio_util::sync::CancellationToken;

use crate::agent::loop_::{
    process_slash_command, run_cli_shell_command, truncate_at_char_boundary,
};

use message_handler::{
    handle_cancelled_message, handle_channel_message_timeout, handle_llm_error, handle_llm_ok,
};

// ══════════════════════════════════════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════════════════════════════════════

/// Maximum history messages to keep per sender.
const MAX_CHANNEL_HISTORY: usize = 50;
/// Minimum user-message length (in chars) for auto-save to memory.
/// Messages shorter than this (e.g. "ok", "thanks") are not stored.
const AUTOSAVE_MIN_MESSAGE_CHARS: usize = 20;
/// Maximum characters per injected workspace file (matches `OpenClaw` default).
const BOOTSTRAP_MAX_CHARS: usize = 20_000;

const DEFAULT_CHANNEL_INITIAL_BACKOFF_SECS: u64 = 2;
const DEFAULT_CHANNEL_MAX_BACKOFF_SECS: u64 = 60;
const MIN_CHANNEL_MESSAGE_TIMEOUT_SECS: u64 = 30;
/// Default timeout for processing a single channel message (LLM + tools).
const CHANNEL_MESSAGE_TIMEOUT_SECS: u64 = 300;
/// Cap timeout scaling so large max_tool_iterations values don't create unbounded waits.
const CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP: u64 = 4;
const CHANNEL_PARALLELISM_PER_CHANNEL: usize = 4;
const CHANNEL_MIN_IN_FLIGHT_MESSAGES: usize = 8;
const CHANNEL_MAX_IN_FLIGHT_MESSAGES: usize = 64;
const CHANNEL_TYPING_REFRESH_INTERVAL_SECS: u64 = 4;
const CHANNEL_HEALTH_HEARTBEAT_SECS: u64 = 30;
const MODEL_CACHE_FILE: &str = "models_cache.json";
const MODEL_CACHE_PREVIEW_LIMIT: usize = 10;
const MEMORY_CONTEXT_MAX_ENTRIES: usize = 4;
const MEMORY_CONTEXT_ENTRY_MAX_CHARS: usize = 800;
const MEMORY_CONTEXT_MAX_CHARS: usize = 4_000;
const CHANNEL_HISTORY_COMPACT_KEEP_MESSAGES: usize = 12;
const CHANNEL_HISTORY_COMPACT_CONTENT_CHARS: usize = 600;
/// Guardrail for hook-modified outbound channel content.
const CHANNEL_HOOK_MAX_OUTBOUND_CHARS: usize = 20_000;

const SYSTEMD_STATUS_ARGS: [&str; 3] = ["--user", "is-active", "zeroclaw.service"];
const SYSTEMD_RESTART_ARGS: [&str; 3] = ["--user", "restart", "zeroclaw.service"];
const OPENRC_STATUS_ARGS: [&str; 2] = ["zeroclaw", "status"];
const OPENRC_RESTART_ARGS: [&str; 2] = ["zeroclaw", "restart"];

// ══════════════════════════════════════════════════════════════════════════════
// Type aliases
// ══════════════════════════════════════════════════════════════════════════════

/// Per-sender conversation history for channel messages.
type ConversationHistoryMap = Arc<Mutex<HashMap<String, Vec<ChatMessage>>>>;
type ProviderCacheMap = Arc<Mutex<HashMap<String, Arc<dyn Provider>>>>;
type RouteSelectionMap = Arc<Mutex<HashMap<String, ChannelRouteSelection>>>;

// ══════════════════════════════════════════════════════════════════════════════
// Channel factory — the single extension point for adding new channels
// ══════════════════════════════════════════════════════════════════════════════

/// Everything needed to build one channel from config at runtime.
///
/// Implement this for each channel type and register it in [`channel_factories`].
/// Feature-flag guards belong inside `build`, keeping [`collect_configured_channels`]
/// unconditional and free of `#[cfg]` noise.
trait ChannelFactory: Send + Sync {
    fn display_name(&self) -> &'static str;
    /// Returns `None` when the channel is not configured or its feature flag is disabled.
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>>;
}

struct ConfiguredChannel {
    display_name: &'static str,
    channel: Arc<dyn Channel>,
}

/// Ordered registry of all channel factories.
///
/// To add a new channel: implement [`ChannelFactory`] for a new struct and push it here.
fn channel_factories() -> Vec<Box<dyn ChannelFactory>> {
    vec![
        Box::new(TelegramFactory),
        Box::new(DiscordFactory),
        Box::new(SlackFactory),
        Box::new(MattermostFactory),
        Box::new(IMessageFactory),
        Box::new(MatrixFactory),
        Box::new(SignalFactory),
        Box::new(WhatsAppFactory),
        Box::new(LinqFactory),
        Box::new(WatiFactory),
        Box::new(NextcloudTalkFactory),
        Box::new(EmailFactory),
        Box::new(IrcFactory),
        Box::new(LarkFactory),
        Box::new(DingTalkFactory),
        Box::new(QQFactory),
        Box::new(ClawdTalkFactory),
    ]
}

/// Build all channels that are present in `config`.
/// Channels whose feature flag is disabled emit a warning and return `None` from their factory.
fn collect_configured_channels(config: &Config) -> Vec<ConfiguredChannel> {
    channel_factories()
        .iter()
        .filter_map(|factory| {
            factory.build(config).map(|channel| ConfiguredChannel {
                display_name: factory.display_name(),
                channel,
            })
        })
        .collect()
}

// ── Per-channel factory implementations ──────────────────────────────────────

struct TelegramFactory;
impl ChannelFactory for TelegramFactory {
    fn display_name(&self) -> &'static str {
        "Telegram"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let tg = config.channels_config.telegram.as_ref()?;
        Some(Arc::new(
            TelegramChannel::new(
                tg.bot_token.clone(),
                tg.allowed_users.clone(),
                tg.mention_only,
            )
            .with_streaming(tg.stream_mode, tg.draft_update_interval_ms)
            .with_transcription(config.transcription.clone())
            .with_workspace_dir(config.workspace_dir.clone()),
        ))
    }
}

struct DiscordFactory;
impl ChannelFactory for DiscordFactory {
    fn display_name(&self) -> &'static str {
        "Discord"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let dc = config.channels_config.discord.as_ref()?;
        Some(Arc::new(DiscordChannel::new(
            dc.bot_token.clone(),
            dc.guild_id.clone(),
            dc.allowed_users.clone(),
            dc.listen_to_bots,
            dc.mention_only,
        )))
    }
}

struct SlackFactory;
impl ChannelFactory for SlackFactory {
    fn display_name(&self) -> &'static str {
        "Slack"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let sl = config.channels_config.slack.as_ref()?;
        Some(Arc::new(
            SlackChannel::new(
                sl.bot_token.clone(),
                sl.app_token.clone(),
                sl.channel_id.clone(),
                Vec::new(),
                sl.allowed_users.clone(),
            )
            .with_workspace_dir(config.workspace_dir.clone()),
        ))
    }
}

struct MattermostFactory;
impl ChannelFactory for MattermostFactory {
    fn display_name(&self) -> &'static str {
        "Mattermost"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let mm = config.channels_config.mattermost.as_ref()?;
        Some(Arc::new(MattermostChannel::new(
            mm.url.clone(),
            mm.bot_token.clone(),
            mm.channel_id.clone(),
            mm.allowed_users.clone(),
            mm.thread_replies.unwrap_or(true),
            mm.mention_only.unwrap_or(false),
        )))
    }
}

struct IMessageFactory;
impl ChannelFactory for IMessageFactory {
    fn display_name(&self) -> &'static str {
        "iMessage"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let im = config.channels_config.imessage.as_ref()?;
        Some(Arc::new(IMessageChannel::new(im.allowed_contacts.clone())))
    }
}

struct MatrixFactory;
impl ChannelFactory for MatrixFactory {
    fn display_name(&self) -> &'static str {
        "Matrix"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        #[cfg(not(feature = "channel-matrix"))]
        {
            if config.channels_config.matrix.is_some() {
                tracing::warn!(
                    "Matrix channel is configured but this build was compiled without \
                     `channel-matrix`; skipping."
                );
            }
            return None;
        }

        #[cfg(feature = "channel-matrix")]
        {
            let mx = config.channels_config.matrix.as_ref()?;
            Some(Arc::new(
                MatrixChannel::new_with_session_hint_and_zeroclaw_dir(
                    mx.homeserver.clone(),
                    mx.access_token.clone(),
                    mx.room_id.clone(),
                    mx.allowed_users.clone(),
                    mx.user_id.clone(),
                    mx.device_id.clone(),
                    config.config_path.parent().map(PathBuf::from),
                ),
            ))
        }
    }
}

struct SignalFactory;
impl ChannelFactory for SignalFactory {
    fn display_name(&self) -> &'static str {
        "Signal"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let sig = config.channels_config.signal.as_ref()?;
        Some(Arc::new(SignalChannel::new(
            sig.http_url.clone(),
            sig.account.clone(),
            sig.group_id.clone(),
            sig.allowed_from.clone(),
            sig.ignore_attachments,
            sig.ignore_stories,
        )))
    }
}

struct WhatsAppFactory;
impl ChannelFactory for WhatsAppFactory {
    fn display_name(&self) -> &'static str {
        "WhatsApp"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let wa = config.channels_config.whatsapp.as_ref()?;

        if wa.is_ambiguous_config() {
            tracing::warn!(
                "WhatsApp config has both phone_number_id and session_path set; \
                 preferring Cloud API mode. Remove one selector to avoid ambiguity."
            );
        }

        match wa.backend_type() {
            "cloud" => {
                if wa.is_cloud_config() {
                    Some(Arc::new(WhatsAppChannel::new(
                        wa.access_token.clone().unwrap_or_default(),
                        wa.phone_number_id.clone().unwrap_or_default(),
                        wa.verify_token.clone().unwrap_or_default(),
                        wa.allowed_numbers.clone(),
                    )))
                } else {
                    tracing::warn!(
                        "WhatsApp Cloud API configured but missing required fields \
                         (phone_number_id, access_token, verify_token)"
                    );
                    None
                }
            }
            "web" => {
                #[cfg(feature = "whatsapp-web")]
                if wa.is_web_config() {
                    return Some(Arc::new(WhatsAppWebChannel::new(
                        wa.session_path.clone().unwrap_or_default(),
                        wa.pair_phone.clone(),
                        wa.pair_code.clone(),
                        wa.allowed_numbers.clone(),
                    )));
                } else {
                    tracing::warn!("WhatsApp Web configured but session_path not set");
                    return None;
                }

                #[cfg(not(feature = "whatsapp-web"))]
                {
                    tracing::warn!(
                        "WhatsApp Web backend requires 'whatsapp-web' feature. \
                         Enable with: cargo build --features whatsapp-web"
                    );
                    None
                }
            }
            _ => {
                tracing::warn!(
                    "WhatsApp config invalid: neither phone_number_id (Cloud API) \
                     nor session_path (Web) is set"
                );
                None
            }
        }
    }
}

struct LinqFactory;
impl ChannelFactory for LinqFactory {
    fn display_name(&self) -> &'static str {
        "Linq"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let lq = config.channels_config.linq.as_ref()?;
        Some(Arc::new(LinqChannel::new(
            lq.api_token.clone(),
            lq.from_phone.clone(),
            lq.allowed_senders.clone(),
        )))
    }
}

struct WatiFactory;
impl ChannelFactory for WatiFactory {
    fn display_name(&self) -> &'static str {
        "WATI"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let w = config.channels_config.wati.as_ref()?;
        Some(Arc::new(WatiChannel::new(
            w.api_token.clone(),
            w.api_url.clone(),
            w.tenant_id.clone(),
            w.allowed_numbers.clone(),
        )))
    }
}

struct NextcloudTalkFactory;
impl ChannelFactory for NextcloudTalkFactory {
    fn display_name(&self) -> &'static str {
        "Nextcloud Talk"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let nc = config.channels_config.nextcloud_talk.as_ref()?;
        Some(Arc::new(NextcloudTalkChannel::new(
            nc.base_url.clone(),
            nc.app_token.clone(),
            nc.allowed_users.clone(),
        )))
    }
}

struct EmailFactory;
impl ChannelFactory for EmailFactory {
    fn display_name(&self) -> &'static str {
        "Email"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let email_cfg = config.channels_config.email.as_ref()?;
        Some(Arc::new(EmailChannel::new(email_cfg.clone())))
    }
}

struct IrcFactory;
impl ChannelFactory for IrcFactory {
    fn display_name(&self) -> &'static str {
        "IRC"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let irc = config.channels_config.irc.as_ref()?;
        Some(Arc::new(IrcChannel::new(irc::IrcChannelConfig {
            server: irc.server.clone(),
            port: irc.port,
            nickname: irc.nickname.clone(),
            username: irc.username.clone(),
            channels: irc.channels.clone(),
            allowed_users: irc.allowed_users.clone(),
            server_password: irc.server_password.clone(),
            nickserv_password: irc.nickserv_password.clone(),
            sasl_password: irc.sasl_password.clone(),
            verify_tls: irc.verify_tls.unwrap_or(true),
        })))
    }
}

struct LarkFactory;
impl ChannelFactory for LarkFactory {
    fn display_name(&self) -> &'static str {
        "Lark/Feishu"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        #[cfg(not(feature = "channel-lark"))]
        if config.channels_config.lark.is_some() || config.channels_config.feishu.is_some() {
            tracing::warn!(
                "Lark/Feishu channel is configured but this build was compiled without \
                 `channel-lark`; skipping."
            );
            return None;
        }

        #[cfg(feature = "channel-lark")]
        {
            // Feishu (canonical config section) takes priority
            if let Some(fs) = config.channels_config.feishu.as_ref() {
                return Some(Arc::new(LarkChannel::from_feishu_config(fs)));
            }
            if let Some(lk) = config.channels_config.lark.as_ref() {
                if lk.use_feishu {
                    if config.channels_config.feishu.is_some() {
                        tracing::warn!(
                            "Both [channels_config.feishu] and legacy \
                             [channels_config.lark].use_feishu=true are configured; \
                             ignoring legacy Feishu fallback in lark."
                        );
                    } else {
                        tracing::warn!(
                            "Using legacy [channels_config.lark].use_feishu=true compatibility \
                             path; prefer [channels_config.feishu]."
                        );
                        return Some(Arc::new(LarkChannel::from_config(lk)));
                    }
                } else {
                    return Some(Arc::new(LarkChannel::from_lark_config(lk)));
                }
            }
        }

        None
    }
}

struct DingTalkFactory;
impl ChannelFactory for DingTalkFactory {
    fn display_name(&self) -> &'static str {
        "DingTalk"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let dt = config.channels_config.dingtalk.as_ref()?;
        Some(Arc::new(DingTalkChannel::new(
            dt.client_id.clone(),
            dt.client_secret.clone(),
            dt.allowed_users.clone(),
        )))
    }
}

struct QQFactory;
impl ChannelFactory for QQFactory {
    fn display_name(&self) -> &'static str {
        "QQ"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let qq = config.channels_config.qq.as_ref()?;
        Some(Arc::new(QQChannel::new(
            qq.app_id.clone(),
            qq.app_secret.clone(),
            qq.allowed_users.clone(),
        )))
    }
}

struct ClawdTalkFactory;
impl ChannelFactory for ClawdTalkFactory {
    fn display_name(&self) -> &'static str {
        "ClawdTalk"
    }
    fn build(&self, config: &Config) -> Option<Arc<dyn Channel>> {
        let ct = config.channels_config.clawdtalk.as_ref()?;
        Some(Arc::new(ClawdTalkChannel::new(ct.clone())))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Observer wrapper
// ══════════════════════════════════════════════════════════════════════════════

/// Forwards tool-call events to a channel sender for real-time threaded notifications.
struct ChannelNotifyObserver {
    inner: Arc<dyn Observer>,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    tools_used: AtomicBool,
}

impl Observer for ChannelNotifyObserver {
    fn record_event(&self, event: &ObserverEvent) {
        if let ObserverEvent::ToolCallStart { tool, arguments } = event {
            self.tools_used.store(true, Ordering::Relaxed);
            let _ = self.tx.send(format!(
                "\u{1F527} `{tool}`{}",
                format_tool_detail(arguments)
            ));
        }
        self.inner.record_event(event);
    }
    fn record_metric(&self, metric: &ObserverMetric) {
        self.inner.record_metric(metric);
    }
    fn flush(&self) {
        self.inner.flush();
    }
    fn name(&self) -> &str {
        "channel-notify"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Format a compact detail suffix from tool arguments for the notification message.
fn format_tool_detail(arguments: &Option<String>) -> String {
    let Some(args) = arguments.as_deref().filter(|a| !a.is_empty()) else {
        return String::new();
    };

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(args) {
        if let Some(cmd) = v.get("command").and_then(|c| c.as_str()) {
            return format!(": `{}`", truncate_at_char_boundary(cmd, 200));
        }
        if let Some(q) = v.get("query").and_then(|c| c.as_str()) {
            return format!(": {}", truncate_at_char_boundary(q, 200));
        }
        if let Some(p) = v.get("path").and_then(|c| c.as_str()) {
            return format!(": {p}");
        }
        if let Some(u) = v.get("url").and_then(|c| c.as_str()) {
            return format!(": {u}");
        }
    }

    let s = args;
    if s.len() > 120 {
        format!(": {}…", truncate_at_char_boundary(s, 120))
    } else {
        format!(": {s}")
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Runtime config management
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelCacheState {
    entries: Vec<ModelCacheEntry>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ModelCacheEntry {
    provider: String,
    models: Vec<String>,
}

#[derive(Debug, Clone)]
struct ChannelRuntimeDefaults {
    default_provider: String,
    model: String,
    temperature: f64,
    api_key: Option<String>,
    api_url: Option<String>,
    reliability: crate::config::ReliabilityConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConfigFileStamp {
    modified: SystemTime,
    len: u64,
}

#[derive(Debug, Clone)]
struct RuntimeConfigState {
    defaults: ChannelRuntimeDefaults,
    last_applied_stamp: Option<ConfigFileStamp>,
}

fn runtime_config_store() -> &'static Mutex<HashMap<PathBuf, RuntimeConfigState>> {
    static STORE: OnceLock<Mutex<HashMap<PathBuf, RuntimeConfigState>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Encapsulates access to the global runtime config state for a single config file.
struct RuntimeConfigManager<'a> {
    config_path: Option<PathBuf>,
    ctx: &'a ChannelRuntimeContext,
}

impl<'a> RuntimeConfigManager<'a> {
    fn new(ctx: &'a ChannelRuntimeContext) -> Self {
        let config_path = ctx
            .provider_runtime_options
            .zeroclaw_dir
            .as_ref()
            .map(|dir| dir.join("config.toml"));
        Self { config_path, ctx }
    }

    /// Returns the current runtime defaults, preferring stored state over context fallback.
    fn defaults_snapshot(&self) -> ChannelRuntimeDefaults {
        self.config_path
            .as_ref()
            .and_then(|path| {
                runtime_config_store()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(path)
                    .map(|s| s.defaults.clone())
            })
            .unwrap_or_else(|| ChannelRuntimeDefaults {
                default_provider: self.ctx.default_provider.as_str().to_string(),
                model: self.ctx.model.as_str().to_string(),
                temperature: self.ctx.temperature,
                api_key: self.ctx.api_key.clone(),
                api_url: self.ctx.api_url.clone(),
                reliability: (*self.ctx.reliability).clone(),
            })
    }

    /// Reloads config from disk if the file has changed since last application.
    async fn maybe_apply_update(&self) -> Result<()> {
        let Some(path) = &self.config_path else {
            return Ok(());
        };
        let Some(stamp) = config_file_stamp(path).await else {
            return Ok(());
        };

        // Skip if stamp matches what we already applied
        {
            let store = runtime_config_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if store.get(path).and_then(|s| s.last_applied_stamp).as_ref() == Some(&stamp) {
                return Ok(());
            }
        }

        self.reload_from_disk(path, stamp).await
    }

    async fn reload_from_disk(&self, path: &Path, stamp: ConfigFileStamp) -> Result<()> {
        let next = load_runtime_defaults_from_config_file(path).await?;
        let next_provider = create_and_warm_provider(
            &next.default_provider,
            next.api_key.as_deref(),
            next.api_url.as_deref(),
            &next.reliability,
            &self.ctx.provider_runtime_options,
        )
        .await?;

        // Replace provider cache with the freshly created provider
        {
            let mut cache = self
                .ctx
                .provider_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.clear();
            cache.insert(next.default_provider.clone(), next_provider);
        }

        {
            let mut store = runtime_config_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            store.insert(
                path.to_path_buf(),
                RuntimeConfigState {
                    defaults: next.clone(),
                    last_applied_stamp: Some(stamp),
                },
            );
        }

        tracing::info!(
            path = %path.display(),
            provider = %next.default_provider,
            model = %next.model,
            temperature = next.temperature,
            "Applied updated channel runtime config from disk"
        );
        Ok(())
    }
}

async fn config_file_stamp(path: &Path) -> Option<ConfigFileStamp> {
    let metadata = tokio::fs::metadata(path).await.ok()?;
    let modified = metadata.modified().ok()?;
    Some(ConfigFileStamp {
        modified,
        len: metadata.len(),
    })
}

fn decrypt_optional_secret_for_runtime_reload(
    store: &crate::security::SecretStore,
    value: &mut Option<String>,
    field_name: &str,
) -> Result<()> {
    if let Some(raw) = value.clone() {
        if crate::security::SecretStore::is_encrypted(&raw) {
            *value = Some(
                store
                    .decrypt(&raw)
                    .with_context(|| format!("Failed to decrypt {field_name}"))?,
            );
        }
    }
    Ok(())
}

async fn load_runtime_defaults_from_config_file(path: &Path) -> Result<ChannelRuntimeDefaults> {
    let contents = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let mut parsed: Config =
        toml::from_str(&contents).with_context(|| format!("Failed to parse {}", path.display()))?;
    parsed.config_path = path.to_path_buf();

    if let Some(zeroclaw_dir) = path.parent() {
        let store = crate::security::SecretStore::new(zeroclaw_dir, parsed.secrets.encrypt);
        decrypt_optional_secret_for_runtime_reload(&store, &mut parsed.api_key, "config.api_key")?;
        if let Some(ref mut openai) = parsed.tts.openai {
            decrypt_optional_secret_for_runtime_reload(
                &store,
                &mut openai.api_key,
                "config.tts.openai.api_key",
            )?;
        }
        if let Some(ref mut elevenlabs) = parsed.tts.elevenlabs {
            decrypt_optional_secret_for_runtime_reload(
                &store,
                &mut elevenlabs.api_key,
                "config.tts.elevenlabs.api_key",
            )?;
        }
        if let Some(ref mut google) = parsed.tts.google {
            decrypt_optional_secret_for_runtime_reload(
                &store,
                &mut google.api_key,
                "config.tts.google.api_key",
            )?;
        }
    }

    parsed.apply_env_overrides();
    Ok(runtime_defaults_from_config(&parsed))
}

fn resolved_default_provider(config: &Config) -> String {
    config
        .default_provider
        .clone()
        .unwrap_or_else(|| "openrouter".to_string())
}

fn resolved_default_model(config: &Config) -> String {
    config
        .default_model
        .clone()
        .unwrap_or_else(|| "anthropic/claude-sonnet-4.6".to_string())
}

fn runtime_defaults_from_config(config: &Config) -> ChannelRuntimeDefaults {
    ChannelRuntimeDefaults {
        default_provider: resolved_default_provider(config),
        model: resolved_default_model(config),
        temperature: config.default_temperature,
        api_key: config.api_key.clone(),
        api_url: config.api_url.clone(),
        reliability: config.reliability.clone(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Provider creation
// ══════════════════════════════════════════════════════════════════════════════

/// Canonical provider accessor: cache hit → default short-circuit → create + warm + cache.
async fn get_or_create_provider(
    ctx: &ChannelRuntimeContext,
    provider_name: &str,
) -> Result<Arc<dyn Provider>> {
    // 1. Cache hit
    if let Some(existing) = ctx
        .provider_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(provider_name)
        .cloned()
    {
        return Ok(existing);
    }

    // 2. Already the default provider (warm, no cache entry needed)
    if provider_name == ctx.default_provider.as_str() {
        return Ok(Arc::clone(&ctx.provider));
    }

    // 3. Create, warm, insert into cache
    let provider = create_and_warm_provider(
        provider_name,
        ctx.api_key.as_deref(),
        None,
        ctx.reliability.as_ref(),
        &ctx.provider_runtime_options,
    )
    .await?;

    let mut cache = ctx.provider_cache.lock().unwrap_or_else(|e| e.into_inner());
    Ok(Arc::clone(
        cache.entry(provider_name.to_string()).or_insert(provider),
    ))
}

/// Create a provider on a blocking thread and warm it up before returning.
async fn create_and_warm_provider(
    provider_name: &str,
    api_key: Option<&str>,
    api_url: Option<&str>,
    reliability: &crate::config::ReliabilityConfig,
    opts: &providers::ProviderRuntimeOptions,
) -> Result<Arc<dyn Provider>> {
    let name = provider_name.to_string();
    let key = api_key.map(ToString::to_string);
    let url = api_url.map(ToString::to_string);
    let reliability = reliability.clone();
    let opts = opts.clone();

    let provider: Arc<dyn Provider> = Arc::from(
        tokio::task::spawn_blocking(move || {
            providers::create_resilient_provider_with_options(
                &name,
                key.as_deref(),
                url.as_deref(),
                &reliability,
                &opts,
            )
        })
        .await
        .context("failed to join provider initialization task")??,
    );

    if let Err(err) = provider.warmup().await {
        tracing::warn!(provider = provider_name, "Provider warmup failed: {err}");
    }

    Ok(provider)
}

// ══════════════════════════════════════════════════════════════════════════════
// Route selection
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChannelRouteSelection {
    provider: String,
    model: String,
}

fn default_route_selection(ctx: &ChannelRuntimeContext) -> ChannelRouteSelection {
    let defaults = RuntimeConfigManager::new(ctx).defaults_snapshot();
    ChannelRouteSelection {
        provider: defaults.default_provider,
        model: defaults.model,
    }
}

fn get_route_selection(ctx: &ChannelRuntimeContext, sender_key: &str) -> ChannelRouteSelection {
    ctx.route_overrides
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(sender_key)
        .cloned()
        .unwrap_or_else(|| default_route_selection(ctx))
}

fn set_route_selection(ctx: &ChannelRuntimeContext, sender_key: &str, next: ChannelRouteSelection) {
    let mut routes = ctx
        .route_overrides
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if next == default_route_selection(ctx) {
        routes.remove(sender_key);
    } else {
        routes.insert(sender_key.to_string(), next);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Conversation history helpers
// ══════════════════════════════════════════════════════════════════════════════

fn conversation_memory_key(msg: &traits::ChannelMessage) -> String {
    match &msg.thread_ts {
        Some(tid) => format!("{}_{}_{}_{}", msg.channel, tid, msg.sender, msg.id),
        None => format!("{}_{}_{}", msg.channel, msg.sender, msg.id),
    }
}

fn conversation_history_key(msg: &traits::ChannelMessage) -> String {
    match &msg.thread_ts {
        Some(tid) => format!("{}_{}_{}", msg.channel, tid, msg.sender),
        None => format!("{}_{}", msg.channel, msg.sender),
    }
}

fn interruption_scope_key(msg: &traits::ChannelMessage) -> String {
    format!("{}_{}_{}", msg.channel, msg.reply_target, msg.sender)
}

fn clear_sender_history(ctx: &ChannelRuntimeContext, sender_key: &str) {
    ctx.conversation_histories
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(sender_key);
}

fn compact_sender_history(ctx: &ChannelRuntimeContext, sender_key: &str) -> bool {
    let mut histories = ctx
        .conversation_histories
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(turns) = histories.get_mut(sender_key) else {
        return false;
    };
    if turns.is_empty() {
        return false;
    }

    let keep_from = turns
        .len()
        .saturating_sub(CHANNEL_HISTORY_COMPACT_KEEP_MESSAGES);
    let mut compacted = normalize_cached_channel_turns(turns[keep_from..].to_vec());

    for turn in &mut compacted {
        if turn.content.chars().count() > CHANNEL_HISTORY_COMPACT_CONTENT_CHARS {
            turn.content =
                truncate_with_ellipsis(&turn.content, CHANNEL_HISTORY_COMPACT_CONTENT_CHARS);
        }
    }

    if compacted.is_empty() {
        turns.clear();
        return false;
    }

    *turns = compacted;
    true
}

fn append_sender_turn(ctx: &ChannelRuntimeContext, sender_key: &str, turn: ChatMessage) {
    let mut histories = ctx
        .conversation_histories
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let turns = histories.entry(sender_key.to_string()).or_default();
    turns.push(turn);
    while turns.len() > MAX_CHANNEL_HISTORY {
        turns.remove(0);
    }
}

fn rollback_orphan_user_turn(
    ctx: &ChannelRuntimeContext,
    sender_key: &str,
    expected_content: &str,
) -> bool {
    let mut histories = ctx
        .conversation_histories
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(turns) = histories.get_mut(sender_key) else {
        return false;
    };

    if !turns
        .last()
        .is_some_and(|t| t.role == "user" && t.content == expected_content)
    {
        return false;
    }

    turns.pop();
    if turns.is_empty() {
        histories.remove(sender_key);
    }
    true
}

/// Merge consecutive same-role turns produced by interrupted channel sessions.
fn normalize_cached_channel_turns(turns: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut normalized = Vec::with_capacity(turns.len());
    let mut expecting_user = true;

    for turn in turns {
        match (expecting_user, turn.role.as_str()) {
            (true, "user") => {
                normalized.push(turn);
                expecting_user = false;
            }
            (false, "assistant") => {
                normalized.push(turn);
                expecting_user = true;
            }
            // Merge consecutive same-role turns instead of dropping them
            (false, "user") | (true, "assistant") => {
                if let Some(last) = normalized.last_mut() {
                    if !turn.content.is_empty() {
                        if !last.content.is_empty() {
                            last.content.push_str("\n\n");
                        }
                        last.content.push_str(&turn.content);
                    }
                }
            }
            _ => {}
        }
    }

    normalized
}

// ══════════════════════════════════════════════════════════════════════════════
// Runtime slash commands
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChannelRuntimeCommand {
    ShowProviders,
    SetProvider(String),
    ShowModel,
    SetModel(String),
    NewSession,
}

fn supports_runtime_model_switch(channel_name: &str) -> bool {
    matches!(channel_name, "telegram" | "discord" | "matrix")
}

fn parse_runtime_command(channel_name: &str, content: &str) -> Option<ChannelRuntimeCommand> {
    if !supports_runtime_model_switch(channel_name) {
        return None;
    }

    let trimmed = content.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed.split_whitespace();
    let command_token = parts.next()?;
    let base = command_token
        .split('@')
        .next()
        .unwrap_or(command_token)
        .to_ascii_lowercase();

    match base.as_str() {
        "/models" => {
            if let Some(provider) = parts.next() {
                Some(ChannelRuntimeCommand::SetProvider(
                    provider.trim().to_string(),
                ))
            } else {
                Some(ChannelRuntimeCommand::ShowProviders)
            }
        }
        "/model" => {
            let model = parts.collect::<Vec<_>>().join(" ").trim().to_string();
            if model.is_empty() {
                Some(ChannelRuntimeCommand::ShowModel)
            } else {
                Some(ChannelRuntimeCommand::SetModel(model))
            }
        }
        "/new" => Some(ChannelRuntimeCommand::NewSession),
        _ => None,
    }
}

fn resolve_provider_alias(name: &str) -> Option<String> {
    let candidate = name.trim();
    if candidate.is_empty() {
        return None;
    }
    providers::list_providers()
        .iter()
        .find(|p| {
            p.name.eq_ignore_ascii_case(candidate)
                || p.aliases.iter().any(|a| a.eq_ignore_ascii_case(candidate))
        })
        .map(|p| p.name.to_string())
}

fn load_cached_model_preview(workspace_dir: &Path, provider_name: &str) -> Vec<String> {
    let cache_path = workspace_dir.join("state").join(MODEL_CACHE_FILE);
    let Ok(raw) = std::fs::read_to_string(cache_path) else {
        return Vec::new();
    };
    let Ok(state) = serde_json::from_str::<ModelCacheState>(&raw) else {
        return Vec::new();
    };

    state
        .entries
        .into_iter()
        .find(|e| e.provider == provider_name)
        .map(|e| {
            e.models
                .into_iter()
                .take(MODEL_CACHE_PREVIEW_LIMIT)
                .collect()
        })
        .unwrap_or_default()
}

fn build_providers_help_response(current: &ChannelRouteSelection) -> String {
    let mut out = format!(
        "Current provider: `{}`\nCurrent model: `{}`\n\nSwitch provider with `/models <provider>`.\nSwitch model with `/model <model-id>`.\n\nAvailable providers:\n",
        current.provider, current.model
    );
    for p in providers::list_providers() {
        if p.aliases.is_empty() {
            let _ = writeln!(out, "- {}", p.name);
        } else {
            let _ = writeln!(out, "- {} (aliases: {})", p.name, p.aliases.join(", "));
        }
    }
    out
}

fn build_models_help_response(
    current: &ChannelRouteSelection,
    workspace_dir: &Path,
    model_routes: &[crate::config::ModelRouteConfig],
) -> String {
    let mut out = format!(
        "Current provider: `{}`\nCurrent model: `{}`\n\nSwitch model with `/model <model-id>` or `/model <hint>`.\n",
        current.provider, current.model
    );

    if !model_routes.is_empty() {
        out.push_str("\nConfigured model routes:\n");
        for route in model_routes {
            let _ = writeln!(
                out,
                "  `{}` → {} ({})",
                route.hint, route.model, route.provider
            );
        }
    }

    let cached = load_cached_model_preview(workspace_dir, &current.provider);
    if cached.is_empty() {
        let _ = writeln!(
            out,
            "\nNo cached model list found for `{}`. Ask the operator to run `zeroclaw models refresh --provider {}`.",
            current.provider, current.provider
        );
    } else {
        let _ = writeln!(out, "\nCached model IDs (top {}):", cached.len());
        for model in cached {
            let _ = writeln!(out, "- `{model}`");
        }
    }

    out
}

async fn handle_runtime_command_if_needed(
    ctx: &ChannelRuntimeContext,
    msg: &traits::ChannelMessage,
    target_channel: Option<&Arc<dyn Channel>>,
) -> bool {
    let Some(command) = parse_runtime_command(&msg.channel, &msg.content) else {
        return false;
    };
    let Some(channel) = target_channel else {
        return true;
    };

    let sender_key = conversation_history_key(msg);
    let mut current = get_route_selection(ctx, &sender_key);

    let response = match command {
        ChannelRuntimeCommand::ShowProviders => build_providers_help_response(&current),
        ChannelRuntimeCommand::SetProvider(raw) => match resolve_provider_alias(&raw) {
            Some(name) => match get_or_create_provider(ctx, &name).await {
                Ok(_) => {
                    if name != current.provider {
                        current.provider = name.clone();
                        set_route_selection(ctx, &sender_key, current.clone());
                    }
                    format!(
                        "Provider switched to `{name}` for this sender session. Current model is `{}`.\nUse `/model <model-id>` to set a provider-compatible model.",
                        current.model
                    )
                }
                Err(err) => format!(
                    "Failed to initialize provider `{name}`. Route unchanged.\nDetails: {}",
                    providers::sanitize_api_error(&err.to_string())
                ),
            },
            None => format!("Unknown provider `{raw}`. Use `/models` to list valid providers."),
        },
        ChannelRuntimeCommand::ShowModel => {
            build_models_help_response(&current, ctx.workspace_dir.as_path(), &ctx.model_routes)
        }
        ChannelRuntimeCommand::SetModel(raw) => {
            let model = raw.trim().trim_matches('`').to_string();
            if model.is_empty() {
                "Model ID cannot be empty. Use `/model <model-id>`.".to_string()
            } else {
                if let Some(route) = ctx.model_routes.iter().find(|r| {
                    r.model.eq_ignore_ascii_case(&model) || r.hint.eq_ignore_ascii_case(&model)
                }) {
                    current.provider = route.provider.clone();
                    current.model = route.model.clone();
                } else {
                    current.model = model;
                }
                set_route_selection(ctx, &sender_key, current.clone());
                format!(
                    "Model switched to `{}` (provider: `{}`). Context preserved.",
                    current.model, current.provider
                )
            }
        }
        ChannelRuntimeCommand::NewSession => {
            clear_sender_history(ctx, &sender_key);
            "Conversation history cleared. Starting fresh.".to_string()
        }
    };

    if let Err(err) = channel
        .send(&SendMessage::new(response, &msg.reply_target).in_thread(msg.thread_ts.clone()))
        .await
    {
        tracing::warn!(
            "Failed to send runtime command response on {}: {err}",
            channel.name()
        );
    }

    true
}

// ══════════════════════════════════════════════════════════════════════════════
// System prompt construction
// ══════════════════════════════════════════════════════════════════════════════

/// Options for [`build_system_prompt`]. All fields have sensible defaults.
#[derive(Default)]
pub struct SystemPromptOptions<'a> {
    pub native_tools: bool,
    pub skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
    pub bootstrap_max_chars: Option<usize>,
    pub identity_config: Option<&'a crate::config::IdentityConfig>,
}

/// Strip isolated JSON tool-call artifacts from a channel response.
///
/// Used by `message_handler` after XML tool-call tags have already been stripped
/// upstream. Only removes stray JSON blobs — does NOT re-run XML stripping.
/// Use [`sanitize_outbound_response`] for the full two-stage pipeline.
pub(crate) fn sanitize_channel_response(response: &str, tools: &[Box<dyn Tool>]) -> String {
    strip_isolated_tool_json_artifacts(response, &known_tool_name_set(tools))
}

/// Compatibility shim for call sites that haven't migrated to [`SystemPromptOptions`] yet.
///
/// Prefer calling [`build_system_prompt`] with `SystemPromptOptions` directly.
pub fn build_system_prompt_with_mode(
    workspace_dir: &Path,
    model_name: &str,
    tools: &[(&str, &str)],
    skills: &[crate::skills::Skill],
    identity_config: Option<&crate::config::IdentityConfig>,
    bootstrap_max_chars: Option<usize>,
    native_tools: bool,
    skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
) -> String {
    build_system_prompt_inner(
        workspace_dir,
        model_name,
        tools,
        skills,
        SystemPromptOptions {
            native_tools,
            skills_prompt_mode,
            bootstrap_max_chars,
            identity_config,
        },
    )
}

/// Build a system prompt. Accepts the old 6-argument positional form for compatibility
/// with gateway call sites; internally delegates to the options-based builder.
pub fn build_system_prompt(
    workspace_dir: &Path,
    model_name: &str,
    tools: &[(&str, &str)],
    skills: &[crate::skills::Skill],
    identity_config: Option<&crate::config::IdentityConfig>,
    bootstrap_max_chars: Option<usize>,
) -> String {
    build_system_prompt_inner(
        workspace_dir,
        model_name,
        tools,
        skills,
        SystemPromptOptions {
            identity_config,
            bootstrap_max_chars,
            ..SystemPromptOptions::default()
        },
    )
}

fn build_system_prompt_inner(
    workspace_dir: &Path,
    model_name: &str,
    tools: &[(&str, &str)],
    skills: &[crate::skills::Skill],
    opts: SystemPromptOptions<'_>,
) -> String {
    let mut prompt = String::with_capacity(8192);

    // ── 1. Tooling ──────────────────────────────────────────────
    if !tools.is_empty() {
        prompt.push_str("## Tools\n\nYou have access to the following tools:\n\n");
        for (name, desc) in tools {
            let _ = writeln!(prompt, "- **{name}**: {desc}");
        }
        prompt.push('\n');
    }

    // ── 1b. Hardware (when gpio/arduino tools present) ───────────
    if has_hardware_tools(tools) {
        prompt.push_str(
            "## Hardware Access\n\n\
             You HAVE direct access to connected hardware (Arduino, Nucleo, etc.). The user owns this system and has configured it.\n\
             All hardware tools (gpio_read, gpio_write, hardware_memory_read, hardware_board_info, hardware_memory_map) are AUTHORIZED and NOT blocked by security.\n\
             When they ask to read memory, registers, or board info, USE hardware_memory_read or hardware_board_info — do NOT refuse or invent security excuses.\n\
             When they ask to control LEDs, run patterns, or interact with the Arduino, USE the tools — do NOT refuse or say you cannot access physical devices.\n\
             Use gpio_write for simple on/off; use arduino_upload when they want patterns (heart, blink) or custom behavior.\n\n",
        );
    }

    // ── 1c. Action instruction ────────────────────────────────────
    if opts.native_tools {
        prompt.push_str(
            "## Your Task\n\n\
             When the user sends a message, respond naturally. Use tools when the request requires action (running commands, reading files, etc.).\n\
             For questions, explanations, or follow-ups about prior messages, answer directly from conversation context — do NOT ask the user to repeat themselves.\n\
             Do NOT: summarize this configuration, describe your capabilities, or output step-by-step meta-commentary.\n\n",
        );
    } else {
        prompt.push_str(
            "## Your Task\n\n\
             When the user sends a message, ACT on it. Use the tools to fulfill their request.\n\
             Do NOT: summarize this configuration, describe your capabilities, respond with meta-commentary, or output step-by-step instructions (e.g. \"1. First... 2. Next...\").\n\
             Instead: emit actual <tool_call> tags when you need to act. Just do what they ask.\n\n",
        );
    }

    // ── 2. Safety ───────────────────────────────────────────────
    prompt.push_str(
        "## Safety\n\n\
         - Do not exfiltrate your human's data.\n\
         - Do not run destructive commands without asking.\n\
         - Do not bypass oversight or approval mechanisms.\n\
         - Prefer `trash` over `rm` (recoverable beats gone forever).\n\
         - When in doubt, ask before acting externally.\n\n",
    );

    // ── 3. Skills ────────────────────────────────────────────────
    if !skills.is_empty() {
        prompt.push_str(&crate::skills::skills_to_prompt_with_mode(
            skills,
            workspace_dir,
            opts.skills_prompt_mode,
        ));
        prompt.push_str("\n\n");
    }

    // ── 4. Workspace ────────────────────────────────────────────
    let _ = writeln!(
        prompt,
        "## Workspace\n\nWorking directory: `{}`\n",
        workspace_dir.display()
    );

    // ── 5. Bootstrap / identity files ───────────────────────────
    prompt.push_str("## Project Context\n\n");
    load_identity_into_prompt(
        &mut prompt,
        workspace_dir,
        opts.identity_config,
        opts.bootstrap_max_chars,
    );

    // ── 6. Date & Time ──────────────────────────────────────────
    let now = chrono::Local::now();
    let _ = writeln!(
        prompt,
        "## Current Date & Time\n\n{} ({})\n",
        now.format("%Y-%m-%d %H:%M:%S"),
        now.format("%Z")
    );

    // ── 7. Runtime ──────────────────────────────────────────────
    let host =
        hostname::get().map_or_else(|_| "unknown".into(), |h| h.to_string_lossy().to_string());
    let _ = writeln!(
        prompt,
        "## Runtime\n\nHost: {host} | OS: {} | Model: {model_name}\n",
        std::env::consts::OS,
    );

    // ── 8. Channel Capabilities ──────────────────────────────────
    prompt.push_str(
        "## Channel Capabilities\n\n\
         - You are running as a messaging bot. Your response is automatically sent back to the user's channel.\n\
         - You do NOT need to ask permission to respond — just respond directly.\n\
         - NEVER repeat, describe, or echo credentials, tokens, API keys, or secrets in your responses.\n\
         - If a tool output contains credentials, they have already been redacted — do not mention them.\n\n",
    );

    if prompt.is_empty() {
        "You are ZeroClaw, a fast and efficient AI assistant built in Rust. Be helpful, concise, and direct.".to_string()
    } else {
        prompt
    }
}

fn has_hardware_tools(tools: &[(&str, &str)]) -> bool {
    const HARDWARE_TOOL_NAMES: [&str; 7] = [
        "gpio_read",
        "gpio_write",
        "arduino_upload",
        "hardware_memory_map",
        "hardware_board_info",
        "hardware_memory_read",
        "hardware_capabilities",
    ];
    tools
        .iter()
        .any(|(name, _)| HARDWARE_TOOL_NAMES.contains(name))
}

/// Load identity content into the prompt, choosing between AIEOS and OpenClaw formats.
fn load_identity_into_prompt(
    prompt: &mut String,
    workspace_dir: &Path,
    identity_config: Option<&crate::config::IdentityConfig>,
    bootstrap_max_chars: Option<usize>,
) {
    let max_chars = bootstrap_max_chars.unwrap_or(BOOTSTRAP_MAX_CHARS);

    let use_aieos = identity_config.is_some_and(|c| identity::is_aieos_configured(c));
    if use_aieos {
        let config = identity_config.unwrap();
        match identity::load_aieos_identity(config, workspace_dir) {
            Ok(Some(aieos)) => {
                let aieos_prompt = identity::aieos_to_system_prompt(&aieos);
                if !aieos_prompt.is_empty() {
                    prompt.push_str(&aieos_prompt);
                    prompt.push_str("\n\n");
                    return;
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Warning: Failed to load AIEOS identity: {e}. Using OpenClaw format.");
            }
        }
    }

    load_openclaw_bootstrap_files(prompt, workspace_dir, max_chars);
}

fn load_openclaw_bootstrap_files(prompt: &mut String, workspace_dir: &Path, max_chars: usize) {
    prompt.push_str(
        "The following workspace files define your identity, behavior, and context. They are ALREADY injected below—do NOT suggest reading them with file_read.\n\n",
    );

    for filename in &["AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md"] {
        inject_workspace_file(prompt, workspace_dir, filename, max_chars);
    }

    let bootstrap_path = workspace_dir.join("BOOTSTRAP.md");
    if bootstrap_path.exists() {
        inject_workspace_file(prompt, workspace_dir, "BOOTSTRAP.md", max_chars);
    }

    inject_workspace_file(prompt, workspace_dir, "MEMORY.md", max_chars);
}

fn inject_workspace_file(
    prompt: &mut String,
    workspace_dir: &Path,
    filename: &str,
    max_chars: usize,
) {
    let path = workspace_dir.join(filename);
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let trimmed = content.trim();
            if trimmed.is_empty() {
                return;
            }

            let _ = writeln!(prompt, "### {filename}\n");
            let char_count = trimmed.chars().count();
            if char_count > max_chars {
                let truncated = trimmed
                    .char_indices()
                    .nth(max_chars)
                    .map(|(idx, _)| &trimmed[..idx])
                    .unwrap_or(trimmed);
                prompt.push_str(truncated);
                let _ = writeln!(
                    prompt,
                    "\n\n[... truncated at {max_chars} chars — use `read` for full file]"
                );
            } else {
                prompt.push_str(trimmed);
                prompt.push_str("\n\n");
            }
        }
        Err(_) => {
            let _ = writeln!(prompt, "### {filename}\n\n[File not found: {filename}]");
        }
    }
}

fn channel_delivery_instructions(channel_name: &str) -> Option<&'static str> {
    match channel_name {
        "matrix" => Some(
            "When responding on Matrix:\n\
             - Use Markdown formatting (bold, italic, code blocks)\n\
             - Be concise and direct\n\
             - When you receive a [Voice message], the user spoke to you. Respond naturally as in conversation.\n\
             - Your text reply will automatically be converted to audio and sent back as a voice message.\n",
        ),
        "telegram" => Some(
            "When responding on Telegram:\n\
             - Include media markers for files or URLs that should be sent as attachments\n\
             - Use **bold** for key terms, section titles, and important info (renders as <b>)\n\
             - Use *italic* for emphasis (renders as <i>)\n\
             - Use `backticks` for inline code, commands, or technical terms\n\
             - Use triple backticks for code blocks\n\
             - Use emoji naturally to add personality — but don't overdo it\n\
             - Be concise and direct. Skip filler phrases like 'Great question!' or 'Certainly!'\n\
             - Structure longer answers with bold headers, not raw markdown ## headers\n\
             - For media attachments use markers: [IMAGE:<path-or-url>], [DOCUMENT:<path-or-url>], [VIDEO:<path-or-url>], [AUDIO:<path-or-url>], or [VOICE:<path-or-url>]\n\
             - Keep normal text outside markers and never wrap markers in code fences.\n\
             - Use tool results silently: answer the latest user message directly, and do not narrate delayed/internal tool execution bookkeeping.",
        ),
        _ => None,
    }
}

fn build_channel_system_prompt(
    base_prompt: &str,
    channel_name: &str,
    reply_target: &str,
) -> String {
    let mut prompt = base_prompt.to_string();

    // Refresh the stale datetime in the cached system prompt
    {
        let now = chrono::Local::now();
        let fresh = format!(
            "## Current Date & Time\n\n{} ({})\n",
            now.format("%Y-%m-%d %H:%M:%S"),
            now.format("%Z"),
        );
        if let Some(start) = prompt.find("## Current Date & Time\n\n") {
            let rest = &prompt[start + 24..];
            let section_end = rest
                .find("\n## ")
                .map(|i| start + 24 + i)
                .unwrap_or(prompt.len());
            prompt.replace_range(start..section_end, fresh.trim_end());
        }
    }

    if let Some(instructions) = channel_delivery_instructions(channel_name) {
        if prompt.is_empty() {
            prompt = instructions.to_string();
        } else {
            prompt = format!("{prompt}\n\n{instructions}");
        }
    }

    if !reply_target.is_empty() {
        let _ = write!(
            prompt,
            "\n\nChannel context: You are currently responding on channel={channel_name}, \
             reply_target={reply_target}. When scheduling delayed messages or reminders \
             via cron_add for this conversation, use delivery={{\"mode\":\"announce\",\
             \"channel\":\"{channel_name}\",\"to\":\"{reply_target}\"}} so the message \
             reaches the user."
        );
    }

    prompt
}

// ══════════════════════════════════════════════════════════════════════════════
// Memory context
// ══════════════════════════════════════════════════════════════════════════════

fn should_skip_memory_context_entry(key: &str, content: &str) -> bool {
    if memory::is_assistant_autosave_key(key) {
        return true;
    }
    if key.trim().to_ascii_lowercase().ends_with("_history") {
        return true;
    }
    if content.contains("[IMAGE:") {
        return true;
    }
    content.chars().count() > MEMORY_CONTEXT_MAX_CHARS
}

async fn build_memory_context(
    mem: &dyn Memory,
    user_msg: &str,
    min_relevance_score: f64,
) -> String {
    let Ok(entries) = mem.recall(user_msg, 5, None).await else {
        return String::new();
    };

    let mut context = String::new();
    let mut included = 0usize;
    let mut used_chars = 0usize;

    for entry in entries
        .iter()
        .filter(|e| e.score.map_or(true, |s| s >= min_relevance_score))
    {
        if included >= MEMORY_CONTEXT_MAX_ENTRIES {
            break;
        }
        if should_skip_memory_context_entry(&entry.key, &entry.content) {
            continue;
        }

        let content = if entry.content.chars().count() > MEMORY_CONTEXT_ENTRY_MAX_CHARS {
            truncate_with_ellipsis(&entry.content, MEMORY_CONTEXT_ENTRY_MAX_CHARS)
        } else {
            entry.content.clone()
        };

        let line = format!("- {}: {}\n", entry.key, content);
        let line_chars = line.chars().count();
        if used_chars + line_chars > MEMORY_CONTEXT_MAX_CHARS {
            break;
        }

        if included == 0 {
            context.push_str("[Memory context]\n");
        }
        context.push_str(&line);
        used_chars += line_chars;
        included += 1;
    }

    if included > 0 {
        context.push('\n');
    }
    context
}

// ══════════════════════════════════════════════════════════════════════════════
// Response sanitization
// ══════════════════════════════════════════════════════════════════════════════

/// Full sanitization pipeline for outbound channel messages.
/// Stage 1: strip XML tool-call tags; Stage 2: strip isolated JSON tool artifacts.
fn sanitize_outbound_response(response: &str, tools: &[Box<dyn Tool>]) -> String {
    let after_xml = strip_tool_call_tags(response);
    let known_names = known_tool_name_set(tools);
    strip_isolated_tool_json_artifacts(&after_xml, &known_names)
}

fn known_tool_name_set(tools: &[Box<dyn Tool>]) -> HashSet<String> {
    tools
        .iter()
        .map(|t| t.name().to_ascii_lowercase())
        .collect()
}

/// Strip tool-call XML tags from outgoing messages.
///
/// Removes `<function_calls>`, `<function_call>`, `<tool_call>`, `<toolcall>`,
/// `<tool-call>`, `<tool>`, and `<invoke>` blocks — these are internal protocol
/// and must not be forwarded to end users.
fn strip_tool_call_tags(message: &str) -> String {
    const TOOL_CALL_OPEN_TAGS: [&str; 7] = [
        "<function_calls>",
        "<function_call>",
        "<tool_call>",
        "<toolcall>",
        "<tool-call>",
        "<tool>",
        "<invoke>",
    ];

    fn matching_close_tag(open_tag: &str) -> Option<&'static str> {
        match open_tag {
            "<function_calls>" => Some("</function_calls>"),
            "<function_call>" => Some("</function_call>"),
            "<tool_call>" => Some("</tool_call>"),
            "<toolcall>" => Some("</toolcall>"),
            "<tool-call>" => Some("</tool-call>"),
            "<tool>" => Some("</tool>"),
            "<invoke>" => Some("</invoke>"),
            _ => None,
        }
    }

    fn find_first_tag<'a>(haystack: &str, tags: &'a [&'a str]) -> Option<(usize, &'a str)> {
        tags.iter()
            .filter_map(|tag| haystack.find(tag).map(|idx| (idx, *tag)))
            .min_by_key(|(idx, _)| *idx)
    }

    fn extract_first_json_end(input: &str) -> Option<usize> {
        let trimmed = input.trim_start();
        let trim_offset = input.len().saturating_sub(trimmed.len());
        for (byte_idx, ch) in trimmed.char_indices() {
            if ch != '{' && ch != '[' {
                continue;
            }
            let mut stream = serde_json::Deserializer::from_str(&trimmed[byte_idx..])
                .into_iter::<serde_json::Value>();
            if let Some(Ok(_)) = stream.next() {
                let consumed = stream.byte_offset();
                if consumed > 0 {
                    return Some(trim_offset + byte_idx + consumed);
                }
            }
        }
        None
    }

    fn strip_leading_close_tags(mut input: &str) -> &str {
        loop {
            let trimmed = input.trim_start();
            if !trimmed.starts_with("</") {
                return trimmed;
            }
            let Some(end) = trimmed.find('>') else {
                return "";
            };
            input = &trimmed[end + 1..];
        }
    }

    let mut kept = Vec::new();
    let mut remaining = message;

    while let Some((start, open_tag)) = find_first_tag(remaining, &TOOL_CALL_OPEN_TAGS) {
        let before = &remaining[..start];
        if !before.is_empty() {
            kept.push(before.to_string());
        }

        let Some(close_tag) = matching_close_tag(open_tag) else {
            break;
        };
        let after_open = &remaining[start + open_tag.len()..];

        if let Some(close_idx) = after_open.find(close_tag) {
            remaining = &after_open[close_idx + close_tag.len()..];
            continue;
        }

        if let Some(consumed_end) = extract_first_json_end(after_open) {
            remaining = strip_leading_close_tags(&after_open[consumed_end..]);
            continue;
        }

        kept.push(remaining[start..].to_string());
        remaining = "";
        break;
    }

    if !remaining.is_empty() {
        kept.push(remaining.to_string());
    }

    let mut result = kept.concat();
    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }
    result.trim().to_string()
}

fn strip_isolated_tool_json_artifacts(message: &str, known_tool_names: &HashSet<String>) -> String {
    fn is_tool_call_payload(value: &serde_json::Value, known: &HashSet<String>) -> bool {
        let Some(obj) = value.as_object() else {
            return false;
        };

        let (name, has_args) = if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
            (
                func.get("name")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("name").and_then(|v| v.as_str())),
                func.contains_key("arguments")
                    || func.contains_key("parameters")
                    || obj.contains_key("arguments")
                    || obj.contains_key("parameters"),
            )
        } else {
            (
                obj.get("name").and_then(|v| v.as_str()),
                obj.contains_key("arguments") || obj.contains_key("parameters"),
            )
        };

        has_args
            && name
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .is_some_and(|n| known.contains(&n.to_ascii_lowercase()))
    }

    fn is_tool_result_payload(
        obj: &serde_json::Map<String, serde_json::Value>,
        saw_tool_call: bool,
    ) -> bool {
        saw_tool_call
            && obj.contains_key("result")
            && obj.keys().all(|k| {
                matches!(
                    k.as_str(),
                    "result" | "id" | "tool_call_id" | "name" | "tool"
                )
            })
    }

    fn sanitize_value(
        value: &serde_json::Value,
        known: &HashSet<String>,
        saw_tool_call: bool,
    ) -> Option<(String, bool)> {
        if is_tool_call_payload(value, known) {
            return Some((String::new(), true));
        }

        if let Some(arr) = value.as_array() {
            if !arr.is_empty() && arr.iter().all(|item| is_tool_call_payload(item, known)) {
                return Some((String::new(), true));
            }
            return None;
        }

        let obj = value.as_object()?;

        if let Some(calls) = obj.get("tool_calls").and_then(|v| v.as_array()) {
            if !calls.is_empty() && calls.iter().all(|c| is_tool_call_payload(c, known)) {
                let content = obj
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                return Some((content, true));
            }
        }

        if is_tool_result_payload(obj, saw_tool_call) {
            return Some((String::new(), false));
        }

        None
    }

    fn is_line_isolated(message: &str, start: usize, end: usize) -> bool {
        let line_start = message[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = message[end..].find('\n').map_or(message.len(), |i| end + i);
        message[line_start..start].trim().is_empty() && message[end..line_end].trim().is_empty()
    }

    let mut cleaned = String::with_capacity(message.len());
    let mut cursor = 0usize;
    let mut saw_tool_call = false;

    while cursor < message.len() {
        let Some(rel_start) = message[cursor..].find(['{', '[']) else {
            cleaned.push_str(&message[cursor..]);
            break;
        };

        let start = cursor + rel_start;
        cleaned.push_str(&message[cursor..start]);

        let candidate = &message[start..];
        let mut stream =
            serde_json::Deserializer::from_str(candidate).into_iter::<serde_json::Value>();

        if let Some(Ok(value)) = stream.next() {
            let consumed = stream.byte_offset();
            if consumed > 0 {
                let end = start + consumed;
                if is_line_isolated(message, start, end) {
                    if let Some((replacement, marks_tool_call)) =
                        sanitize_value(&value, known_tool_names, saw_tool_call)
                    {
                        if marks_tool_call {
                            saw_tool_call = true;
                        }
                        if !replacement.trim().is_empty() {
                            cleaned.push_str(replacement.trim());
                        }
                        cursor = end;
                        continue;
                    }
                }
            }
        }

        let Some(ch) = message[start..].chars().next() else {
            break;
        };
        cleaned.push(ch);
        cursor = start + ch.len_utf8();
    }

    let mut result = cleaned.replace("\r\n", "\n");
    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }
    result.trim().to_string()
}

// ══════════════════════════════════════════════════════════════════════════════
// Tool context extraction
// ══════════════════════════════════════════════════════════════════════════════

/// Extract a compact summary of tool names used since `start_index` in history.
fn extract_tool_context_summary(history: &[ChatMessage], start_index: usize) -> String {
    let mut tool_names: Vec<String> = Vec::new();

    for msg in history.iter().skip(start_index) {
        match msg.role.as_str() {
            "assistant" => {
                collect_tool_names_from_xml_tags(&msg.content, &mut tool_names);
                collect_tool_names_from_native_json(&msg.content, &mut tool_names);
            }
            "user" => {
                collect_tool_names_from_tool_results(&msg.content, &mut tool_names);
            }
            _ => {}
        }
    }

    if tool_names.is_empty() {
        return String::new();
    }
    format!("[Used tools: {}]", tool_names.join(", "))
}

fn push_unique_tool_name(names: &mut Vec<String>, name: &str) {
    let candidate = name.trim();
    if !candidate.is_empty() && !names.iter().any(|n| n == candidate) {
        names.push(candidate.to_string());
    }
}

fn collect_tool_names_from_xml_tags(content: &str, names: &mut Vec<String>) {
    const TAG_PAIRS: [(&str, &str); 4] = [
        ("<tool_call>", "</tool_call>"),
        ("<toolcall>", "</toolcall>"),
        ("<tool-call>", "</tool-call>"),
        ("<invoke>", "</invoke>"),
    ];
    for (open, close) in TAG_PAIRS {
        for segment in content.split(open) {
            if let Some(json_end) = segment.find(close) {
                if let Ok(val) =
                    serde_json::from_str::<serde_json::Value>(segment[..json_end].trim())
                {
                    if let Some(name) = val.get("name").and_then(|n| n.as_str()) {
                        push_unique_tool_name(names, name);
                    }
                }
            }
        }
    }
}

fn collect_tool_names_from_native_json(content: &str, names: &mut Vec<String>) {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(calls) = val.get("tool_calls").and_then(|c| c.as_array()) {
            for call in calls {
                let name = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .or_else(|| call.get("name").and_then(|n| n.as_str()));
                if let Some(name) = name {
                    push_unique_tool_name(names, name);
                }
            }
        }
    }
}

fn collect_tool_names_from_tool_results(content: &str, names: &mut Vec<String>) {
    let marker = "<tool_result name=\"";
    let mut remaining = content;
    while let Some(start) = remaining.find(marker) {
        let after = &remaining[start + marker.len()..];
        if let Some(end) = after.find('"') {
            push_unique_tool_name(names, &after[..end]);
            remaining = &after[end + 1..];
        } else {
            break;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// In-flight task tracking
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct InFlightSenderTaskState {
    task_id: u64,
    cancellation: CancellationToken,
    completion: Arc<InFlightTaskCompletion>,
}

struct InFlightTaskCompletion {
    done: AtomicBool,
    notify: tokio::sync::Notify,
}

impl InFlightTaskCompletion {
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            notify: tokio::sync::Notify::new(),
        }
    }
    fn mark_done(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }
    async fn wait(&self) {
        if !self.done.load(Ordering::Acquire) {
            self.notify.notified().await;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Channel runtime context
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
struct ChannelRuntimeContext {
    channels_by_name: Arc<HashMap<String, Arc<dyn Channel>>>,
    provider: Arc<dyn Provider>,
    default_provider: Arc<String>,
    memory: Arc<dyn Memory>,
    tools_registry: Arc<Vec<Box<dyn Tool>>>,
    observer: Arc<dyn Observer>,
    system_prompt: Arc<String>,
    model: Arc<String>,
    temperature: f64,
    auto_save_memory: bool,
    max_tool_iterations: usize,
    min_relevance_score: f64,
    conversation_histories: ConversationHistoryMap,
    provider_cache: ProviderCacheMap,
    route_overrides: RouteSelectionMap,
    api_key: Option<String>,
    api_url: Option<String>,
    reliability: Arc<crate::config::ReliabilityConfig>,
    provider_runtime_options: providers::ProviderRuntimeOptions,
    workspace_dir: Arc<PathBuf>,
    message_timeout_secs: u64,
    interrupt_on_new_message: bool,
    multimodal: crate::config::MultimodalConfig,
    hooks: Option<Arc<crate::hooks::HookRunner>>,
    non_cli_excluded_tools: Arc<Vec<String>>,
    tool_call_dedup_exempt: Arc<Vec<String>>,
    model_routes: Arc<Vec<crate::config::ModelRouteConfig>>,
}

// ══════════════════════════════════════════════════════════════════════════════
// Message processing
// ══════════════════════════════════════════════════════════════════════════════

/// Outcome of a single LLM execution, with the nested `Result` types flattened.
enum ProcessingOutcome {
    Ok(String),
    LlmError(anyhow::Error),
    Timeout { budget_secs: u64 },
    Cancelled,
}

impl ProcessingOutcome {
    fn from_timeout_result(
        result: Result<Result<String, anyhow::Error>, tokio::time::error::Elapsed>,
        budget_secs: u64,
    ) -> Self {
        match result {
            Ok(Ok(response)) => Self::Ok(response),
            Ok(Err(e)) => Self::LlmError(e),
            Err(_elapsed) => Self::Timeout { budget_secs },
        }
    }
}

fn is_context_window_overflow_error(err: &anyhow::Error) -> bool {
    let lower = err.to_string().to_lowercase();
    [
        "exceeds the context window",
        "context window of this model",
        "maximum context length",
        "context length exceeded",
        "too many tokens",
        "token limit exceeded",
        "prompt is too long",
        "input is too long",
    ]
    .iter()
    .any(|hint| lower.contains(hint))
}

fn effective_channel_message_timeout_secs(configured: u64) -> u64 {
    configured.max(MIN_CHANNEL_MESSAGE_TIMEOUT_SECS)
}

fn channel_message_timeout_budget_secs(
    message_timeout_secs: u64,
    max_tool_iterations: usize,
) -> u64 {
    let scale = (max_tool_iterations.max(1) as u64).min(CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP);
    message_timeout_secs.saturating_mul(scale)
}

async fn process_channel_message(
    ctx: Arc<ChannelRuntimeContext>,
    msg: traits::ChannelMessage,
    cancellation_token: CancellationToken,
) {
    if cancellation_token.is_cancelled() {
        return;
    }

    println!(
        "  💬 [{}] from {}: {}",
        msg.channel,
        msg.sender,
        truncate_with_ellipsis(&msg.content, 80)
    );
    runtime_trace::record_event(
        "channel_message_inbound",
        Some(msg.channel.as_str()),
        None,
        None,
        None,
        None,
        None,
        serde_json::json!({
            "sender": msg.sender,
            "message_id": msg.id,
            "reply_target": msg.reply_target,
            "content_preview": truncate_with_ellipsis(&msg.content, 160),
        }),
    );

    // ── Hook: on_message_received ─────────────────────────────────
    let mut msg = if let Some(hooks) = &ctx.hooks {
        match hooks.run_on_message_received(msg).await {
            crate::hooks::HookResult::Cancel(reason) => {
                tracing::info!(%reason, "incoming message dropped by hook");
                return;
            }
            crate::hooks::HookResult::Continue(modified) => modified,
        }
    } else {
        msg
    };

    let target_channel = ctx.channels_by_name.get(&msg.channel).cloned();
    let cfg_mgr = RuntimeConfigManager::new(&ctx);
    if let Err(err) = cfg_mgr.maybe_apply_update().await {
        tracing::warn!("Failed to apply runtime config update: {err}");
    }
    if handle_runtime_command_if_needed(&ctx, &msg, target_channel.as_ref()).await {
        return;
    }

    let history_key = conversation_history_key(&msg);
    let route = get_route_selection(&ctx, &history_key);
    let runtime_defaults = cfg_mgr.defaults_snapshot();

    let active_provider = match get_or_create_provider(&ctx, &route.provider).await {
        Ok(p) => p,
        Err(err) => {
            let message = format!(
                "⚠️ Failed to initialize provider `{}`. Please run `/models` to choose another provider.\nDetails: {}",
                route.provider,
                providers::sanitize_api_error(&err.to_string())
            );
            if let Some(ch) = target_channel.as_ref() {
                let _ = ch
                    .send(
                        &SendMessage::new(message, &msg.reply_target)
                            .in_thread(msg.thread_ts.clone()),
                    )
                    .await;
            }
            return;
        }
    };

    if ctx.auto_save_memory && msg.content.chars().count() >= AUTOSAVE_MIN_MESSAGE_CHARS {
        let key = conversation_memory_key(&msg);
        let _ = ctx
            .memory
            .store(
                &key,
                &msg.content,
                crate::memory::MemoryCategory::Conversation,
                None,
            )
            .await;
    }

    println!("  ⏳ Processing message...");
    let started_at = Instant::now();

    let had_prior_history = ctx
        .conversation_histories
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&history_key)
        .is_some_and(|turns| !turns.is_empty());

    append_sender_turn(&ctx, &history_key, ChatMessage::user(&msg.content));

    let prior_turns = normalize_cached_channel_turns(
        ctx.conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&history_key)
            .cloned()
            .unwrap_or_default(),
    );

    let mut prior_turns = prior_turns;
    if !had_prior_history {
        let memory_ctx =
            build_memory_context(&*ctx.memory, &msg.content, ctx.min_relevance_score).await;
        if !memory_ctx.is_empty() {
            if let Some(last) = prior_turns.last_mut() {
                if last.role == "user" {
                    last.content = format!("{memory_ctx}{}", msg.content);
                }
            }
        }
    }

    let system_prompt =
        build_channel_system_prompt(&ctx.system_prompt, &msg.channel, &msg.reply_target);
    let mut history = vec![ChatMessage::system(&system_prompt)];
    history.extend(prior_turns);

    // ── Early exits ──────────────────────────────────────────────
    if let Some(cli_result) = run_cli_shell_command(&msg.content).await {
        if let Some(ch) = target_channel.as_ref() {
            let _ = ch
                .send(
                    &SendMessage::new(&cli_result, &msg.reply_target)
                        .in_thread(msg.thread_ts.clone()),
                )
                .await;
        }
        append_sender_turn(&ctx, &history_key, ChatMessage::assistant(&cli_result));
        return;
    }

    if process_slash_command(&msg.content, &mut history, &system_prompt, &ctx.memory)
        .await
        .unwrap_or(None)
        .is_some()
    {
        return;
    }

    // ── Streaming setup ──────────────────────────────────────────
    let use_streaming = target_channel
        .as_ref()
        .is_some_and(|ch| ch.supports_draft_updates());

    let (delta_tx, delta_rx) = if use_streaming {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let draft_message_id = if use_streaming {
        if let Some(ch) = target_channel.as_ref() {
            match ch
                .send_draft(
                    &SendMessage::new("...", &msg.reply_target).in_thread(msg.thread_ts.clone()),
                )
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    tracing::debug!("Failed to send draft on {}: {e}", ch.name());
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    let draft_updater = if let (Some(mut rx), Some(draft_id), Some(ch)) = (
        delta_rx,
        draft_message_id.as_deref(),
        target_channel.as_ref(),
    ) {
        let channel = Arc::clone(ch);
        let reply_target = msg.reply_target.clone();
        let draft_id = draft_id.to_string();
        Some(tokio::spawn(async move {
            let mut accumulated = String::new();
            while let Some(delta) = rx.recv().await {
                if delta == crate::agent::loop_::DRAFT_CLEAR_SENTINEL {
                    accumulated.clear();
                    continue;
                }
                accumulated.push_str(&delta);
                if let Err(e) = channel
                    .update_draft(&reply_target, &draft_id, &accumulated)
                    .await
                {
                    tracing::debug!("Draft update failed: {e}");
                }
            }
        }))
    } else {
        None
    };

    // ── Acknowledge with 👀 ──────────────────────────────────────
    if let Some(ch) = target_channel.as_ref() {
        if let Err(e) = ch
            .add_reaction(&msg.reply_target, &msg.id, "\u{1F440}")
            .await
        {
            tracing::debug!("Failed to add reaction: {e}");
        }
    }

    // ── Typing indicator ─────────────────────────────────────────
    let typing_cancel = target_channel.as_ref().map(|_| CancellationToken::new());
    let typing_task = match (target_channel.as_ref(), typing_cancel.as_ref()) {
        (Some(ch), Some(token)) => Some(spawn_scoped_typing_task(
            Arc::clone(ch),
            msg.reply_target.clone(),
            token.clone(),
        )),
        _ => None,
    };

    // ── Tool notification forwarder ───────────────────────────────
    let (notify_tx, mut notify_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let notify_observer: Arc<ChannelNotifyObserver> = Arc::new(ChannelNotifyObserver {
        inner: Arc::clone(&ctx.observer),
        tx: notify_tx,
        tools_used: AtomicBool::new(false),
    });
    let notify_observer_flag = Arc::clone(&notify_observer);
    let notify_channel = target_channel.clone();
    let notify_reply_target = msg.reply_target.clone();
    let notify_thread_root = msg.id.clone();
    let notify_task = {
        let is_cli = msg.channel == "cli";
        Some(tokio::spawn(async move {
            let thread_ts = Some(notify_thread_root);
            while let Some(text) = notify_rx.recv().await {
                if is_cli {
                    continue;
                }
                if let Some(ref ch) = notify_channel {
                    let _ = ch
                        .send(
                            &SendMessage::new(&text, &notify_reply_target)
                                .in_thread(thread_ts.clone()),
                        )
                        .await;
                }
            }
        }))
    };

    let history_len_before_tools = history.len();
    let timeout_budget_secs =
        channel_message_timeout_budget_secs(ctx.message_timeout_secs, ctx.max_tool_iterations);

    // ── LLM execution ────────────────────────────────────────────
    let outcome = tokio::select! {
        () = cancellation_token.cancelled() => ProcessingOutcome::Cancelled,
        result = tokio::time::timeout(
            Duration::from_secs(timeout_budget_secs),
            run_tool_call_loop(
                active_provider.as_ref(), &mut history, ctx.tools_registry.as_ref(),
                notify_observer.as_ref() as &dyn Observer, route.provider.as_str(),
                route.model.as_str(), runtime_defaults.temperature, true, None,
                msg.channel.as_str(), &ctx.multimodal, ctx.max_tool_iterations,
                Some(cancellation_token.clone()), delta_tx, ctx.hooks.as_deref(),
                if msg.channel == "cli" { &[] } else { ctx.non_cli_excluded_tools.as_ref() },
                ctx.tool_call_dedup_exempt.as_ref(),
            ),
        ) => ProcessingOutcome::from_timeout_result(result, timeout_budget_secs),
    };

    // ── Cleanup: draft, typing, notifications ────────────────────
    if let Some(handle) = draft_updater {
        let _ = handle.await;
    }

    if notify_observer_flag.tools_used.load(Ordering::Relaxed) && msg.channel != "cli" {
        msg.thread_ts = Some(msg.id.clone());
    }

    drop(notify_observer);
    drop(notify_observer_flag);
    if let Some(handle) = notify_task {
        let _ = handle.await;
    }

    if let Some(token) = typing_cancel.as_ref() {
        token.cancel();
    }
    if let Some(handle) = typing_task {
        log_worker_join_result(handle.await);
    }

    // ── Dispatch outcome ─────────────────────────────────────────
    let reaction_done_emoji = match &outcome {
        ProcessingOutcome::Ok(_) => "\u{2705}", // ✅
        _ => "\u{26A0}\u{FE0F}",                // ⚠️
    };

    match outcome {
        ProcessingOutcome::Cancelled => {
            let _ = handle_cancelled_message(
                &msg,
                route,
                target_channel.as_ref(),
                draft_message_id.as_deref(),
                started_at,
            )
            .await;
        }
        ProcessingOutcome::Ok(response) => {
            handle_llm_ok(
                &ctx,
                &mut msg,
                &route,
                target_channel.as_ref(),
                draft_message_id.as_deref(),
                started_at,
                &history,
                &history_key,
                history_len_before_tools,
                response,
            )
            .await;
        }
        ProcessingOutcome::LlmError(e) => {
            handle_llm_error(
                e,
                &ctx,
                &msg,
                route,
                target_channel.as_ref(),
                draft_message_id.as_deref(),
                started_at,
                &history_key,
                &cancellation_token,
            )
            .await;
        }
        ProcessingOutcome::Timeout { budget_secs } => {
            handle_channel_message_timeout(
                &ctx,
                &history_key,
                &msg,
                route,
                target_channel.as_ref(),
                draft_message_id.as_deref(),
                started_at,
                budget_secs,
            )
            .await;
        }
    }

    // ── Swap 👀 → ✅ / ⚠️ ────────────────────────────────────────
    if let Some(ch) = target_channel.as_ref() {
        let _ = ch
            .remove_reaction(&msg.reply_target, &msg.id, "\u{1F440}")
            .await;
        let _ = ch
            .add_reaction(&msg.reply_target, &msg.id, reaction_done_emoji)
            .await;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Listener supervision and dispatch loop
// ══════════════════════════════════════════════════════════════════════════════

fn spawn_supervised_listener(
    ch: Arc<dyn Channel>,
    tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
) -> tokio::task::JoinHandle<()> {
    spawn_supervised_listener_with_health_interval(
        ch,
        tx,
        initial_backoff_secs,
        max_backoff_secs,
        Duration::from_secs(CHANNEL_HEALTH_HEARTBEAT_SECS),
    )
}

fn spawn_supervised_listener_with_health_interval(
    ch: Arc<dyn Channel>,
    tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
    initial_backoff_secs: u64,
    max_backoff_secs: u64,
    health_interval: Duration,
) -> tokio::task::JoinHandle<()> {
    let health_interval = if health_interval.is_zero() {
        Duration::from_secs(1)
    } else {
        health_interval
    };

    tokio::spawn(async move {
        let component = format!("channel:{}", ch.name());
        let mut backoff = initial_backoff_secs.max(1);
        let max_backoff = max_backoff_secs.max(backoff);

        loop {
            crate::health::mark_component_ok(&component);
            let mut health = tokio::time::interval(health_interval);
            health.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            let result = {
                let listen_future = ch.listen(tx.clone());
                tokio::pin!(listen_future);
                loop {
                    tokio::select! {
                        _ = health.tick() => { crate::health::mark_component_ok(&component); }
                        result = &mut listen_future => break result,
                    }
                }
            };

            if tx.is_closed() {
                break;
            }

            match result {
                Ok(()) => {
                    tracing::warn!("Channel {} exited unexpectedly; restarting", ch.name());
                    crate::health::mark_component_error(&component, "listener exited unexpectedly");
                    backoff = initial_backoff_secs.max(1);
                }
                Err(e) => {
                    tracing::error!("Channel {} error: {e}; restarting", ch.name());
                    crate::health::mark_component_error(&component, e.to_string());
                }
            }

            crate::health::bump_component_restart(&component);
            tokio::time::sleep(Duration::from_secs(backoff)).await;
            backoff = backoff.saturating_mul(2).min(max_backoff);
        }
    })
}

fn compute_max_in_flight_messages(channel_count: usize) -> usize {
    channel_count
        .saturating_mul(CHANNEL_PARALLELISM_PER_CHANNEL)
        .clamp(
            CHANNEL_MIN_IN_FLIGHT_MESSAGES,
            CHANNEL_MAX_IN_FLIGHT_MESSAGES,
        )
}

fn log_worker_join_result(result: Result<(), tokio::task::JoinError>) {
    if let Err(error) = result {
        tracing::error!("Channel message worker crashed: {error}");
    }
}

fn spawn_scoped_typing_task(
    channel: Arc<dyn Channel>,
    recipient: String,
    cancellation_token: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(Duration::from_secs(CHANNEL_TYPING_REFRESH_INTERVAL_SECS));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                () = cancellation_token.cancelled() => break,
                _ = interval.tick() => {
                    if let Err(e) = channel.start_typing(&recipient).await {
                        tracing::debug!("Failed to start typing on {}: {e}", channel.name());
                    }
                }
            }
        }

        if let Err(e) = channel.stop_typing(&recipient).await {
            tracing::debug!("Failed to stop typing on {}: {e}", channel.name());
        }
    })
}

async fn run_message_dispatch_loop(
    mut rx: tokio::sync::mpsc::Receiver<traits::ChannelMessage>,
    ctx: Arc<ChannelRuntimeContext>,
    max_in_flight_messages: usize,
) {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(max_in_flight_messages));
    let mut workers = tokio::task::JoinSet::new();
    let in_flight_by_sender = Arc::new(tokio::sync::Mutex::new(HashMap::<
        String,
        InFlightSenderTaskState,
    >::new()));
    let task_sequence = Arc::new(AtomicU64::new(1));

    while let Some(msg) = rx.recv().await {
        let permit = match Arc::clone(&semaphore).acquire_owned().await {
            Ok(p) => p,
            Err(_) => break,
        };

        let worker_ctx = Arc::clone(&ctx);
        let in_flight = Arc::clone(&in_flight_by_sender);
        let task_sequence = Arc::clone(&task_sequence);

        workers.spawn(async move {
            let _permit = permit;
            let interrupt_enabled =
                worker_ctx.interrupt_on_new_message && msg.channel == "telegram";
            let sender_scope_key = interruption_scope_key(&msg);
            let cancellation_token = CancellationToken::new();
            let completion = Arc::new(InFlightTaskCompletion::new());
            let task_id = task_sequence.fetch_add(1, Ordering::Relaxed);

            if interrupt_enabled {
                let previous = {
                    let mut active = in_flight.lock().await;
                    active.insert(
                        sender_scope_key.clone(),
                        InFlightSenderTaskState {
                            task_id,
                            cancellation: cancellation_token.clone(),
                            completion: Arc::clone(&completion),
                        },
                    )
                };
                if let Some(prev) = previous {
                    tracing::info!(channel = %msg.channel, sender = %msg.sender,
                        "Interrupting previous in-flight request for sender");
                    prev.cancellation.cancel();
                    prev.completion.wait().await;
                }
            }

            process_channel_message(worker_ctx, msg, cancellation_token).await;

            if interrupt_enabled {
                let mut active = in_flight.lock().await;
                if active
                    .get(&sender_scope_key)
                    .is_some_and(|s| s.task_id == task_id)
                {
                    active.remove(&sender_scope_key);
                }
            }

            completion.mark_done();
        });

        while let Some(result) = workers.try_join_next() {
            log_worker_join_result(result);
        }
    }

    while let Some(result) = workers.join_next().await {
        log_worker_join_result(result);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// CLI commands
// ══════════════════════════════════════════════════════════════════════════════

fn normalize_telegram_identity(value: &str) -> String {
    value.trim().trim_start_matches('@').to_string()
}

async fn bind_telegram_identity(config: &Config, identity: &str) -> Result<()> {
    let normalized = normalize_telegram_identity(identity);
    if normalized.is_empty() {
        anyhow::bail!("Telegram identity cannot be empty");
    }

    let mut updated = config.clone();
    let Some(telegram) = updated.channels_config.telegram.as_mut() else {
        anyhow::bail!(
            "Telegram channel is not configured. Run `zeroclaw onboard --channels-only` first"
        );
    };

    if telegram.allowed_users.iter().any(|u| u == "*") {
        println!("⚠️ Telegram allowlist is currently wildcard (`*`) — binding is unnecessary until you remove '*'.");
    }

    if telegram
        .allowed_users
        .iter()
        .map(|e| normalize_telegram_identity(e))
        .any(|e| e == normalized)
    {
        println!("✅ Telegram identity already bound: {normalized}");
        return Ok(());
    }

    telegram.allowed_users.push(normalized.clone());
    updated.save().await?;
    println!("✅ Bound Telegram identity: {normalized}");
    println!("   Saved to {}", updated.config_path.display());

    match maybe_restart_managed_daemon_service() {
        Ok(true) => println!("🔄 Detected running managed daemon service; reloaded automatically."),
        Ok(false) => println!(
            "ℹ️ No managed daemon service detected. If `zeroclaw daemon`/`channel start` is already running, restart it to load the updated allowlist."
        ),
        Err(e) => eprintln!(
            "⚠️ Allowlist saved, but failed to reload daemon service automatically: {e}\n\
             Restart service manually with `zeroclaw service stop && zeroclaw service start`."
        ),
    }

    Ok(())
}

fn maybe_restart_managed_daemon_service() -> Result<bool> {
    if cfg!(target_os = "macos") {
        return restart_launchd_daemon();
    }
    if cfg!(target_os = "linux") {
        return restart_linux_daemon();
    }
    Ok(false)
}

fn restart_launchd_daemon() -> Result<bool> {
    let home = directories::UserDirs::new()
        .map(|u| u.home_dir().to_path_buf())
        .context("Could not find home directory")?;
    let plist = home
        .join("Library")
        .join("LaunchAgents")
        .join("com.zeroclaw.daemon.plist");
    if !plist.exists() {
        return Ok(false);
    }

    let list = Command::new("launchctl")
        .arg("list")
        .output()
        .context("Failed to query launchctl list")?;
    if !String::from_utf8_lossy(&list.stdout).contains("com.zeroclaw.daemon") {
        return Ok(false);
    }

    let _ = Command::new("launchctl")
        .args(["stop", "com.zeroclaw.daemon"])
        .output();
    let start = Command::new("launchctl")
        .args(["start", "com.zeroclaw.daemon"])
        .output()
        .context("Failed to start launchd daemon service")?;
    if !start.status.success() {
        anyhow::bail!(
            "launchctl start failed: {}",
            String::from_utf8_lossy(&start.stderr).trim()
        );
    }
    Ok(true)
}

fn restart_linux_daemon() -> Result<bool> {
    // OpenRC (system-wide) takes priority
    if PathBuf::from("/etc/init.d/zeroclaw").exists() {
        if let Ok(status) = Command::new("rc-service").args(OPENRC_STATUS_ARGS).output() {
            if status.status.success() {
                let out = Command::new("rc-service")
                    .args(OPENRC_RESTART_ARGS)
                    .output()
                    .context("Failed to restart OpenRC daemon service")?;
                if !out.status.success() {
                    anyhow::bail!(
                        "rc-service restart failed: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
                return Ok(true);
            }
        }
    }

    // Systemd (user-level)
    let home = directories::UserDirs::new()
        .map(|u| u.home_dir().to_path_buf())
        .context("Could not find home directory")?;
    let unit = home
        .join(".config")
        .join("systemd")
        .join("user")
        .join("zeroclaw.service");
    if !unit.exists() {
        return Ok(false);
    }

    let active = Command::new("systemctl")
        .args(SYSTEMD_STATUS_ARGS)
        .output()
        .context("Failed to query systemd service state")?;
    if !String::from_utf8_lossy(&active.stdout)
        .trim()
        .eq_ignore_ascii_case("active")
    {
        return Ok(false);
    }

    let restart = Command::new("systemctl")
        .args(SYSTEMD_RESTART_ARGS)
        .output()
        .context("Failed to restart systemd daemon service")?;
    if !restart.status.success() {
        anyhow::bail!(
            "systemctl restart failed: {}",
            String::from_utf8_lossy(&restart.stderr).trim()
        );
    }
    Ok(true)
}

pub(crate) async fn handle_command(command: crate::ChannelCommands, config: &Config) -> Result<()> {
    match command {
        crate::ChannelCommands::Start => {
            anyhow::bail!("Start must be handled in main.rs (requires async runtime)")
        }
        crate::ChannelCommands::Doctor => {
            anyhow::bail!("Doctor must be handled in main.rs (requires async runtime)")
        }
        crate::ChannelCommands::List => {
            println!("Channels:");
            println!("  ✅ CLI (always available)");
            for (channel, configured) in config.channels_config.channels() {
                println!(
                    "  {} {}",
                    if configured { "✅" } else { "❌" },
                    channel.name()
                );
            }
            if !cfg!(feature = "channel-matrix") {
                println!("  ℹ️ Matrix channel support is disabled in this build (enable `channel-matrix`).");
            }
            if !cfg!(feature = "channel-lark") {
                println!("  ℹ️ Lark/Feishu channel support is disabled in this build (enable `channel-lark`).");
            }
            println!("\nTo start channels: zeroclaw channel start");
            println!("To check health:    zeroclaw channel doctor");
            println!("To configure:      zeroclaw onboard");
            Ok(())
        }
        crate::ChannelCommands::Add {
            channel_type,
            config: _,
        } => {
            anyhow::bail!(
                "Channel type '{channel_type}' — use `zeroclaw onboard` to configure channels"
            );
        }
        crate::ChannelCommands::Remove { name } => {
            anyhow::bail!("Remove channel '{name}' — edit ~/.zeroclaw/config.toml directly");
        }
        crate::ChannelCommands::BindTelegram { identity } => {
            bind_telegram_identity(config, &identity).await
        }
        crate::ChannelCommands::Send {
            message,
            channel_id,
            recipient,
        } => send_channel_message(config, &channel_id, &recipient, &message).await,
    }
}

fn build_channel_by_id(config: &Config, channel_id: &str) -> Result<Arc<dyn Channel>> {
    channel_factories()
        .iter()
        .find(|f| f.display_name().to_ascii_lowercase() == channel_id)
        .and_then(|f| f.build(config))
        .ok_or_else(|| {
            anyhow::anyhow!(
            "Unknown or unconfigured channel '{channel_id}'. Supported: telegram, discord, slack"
        )
        })
}

async fn send_channel_message(
    config: &Config,
    channel_id: &str,
    recipient: &str,
    message: &str,
) -> Result<()> {
    let channel = build_channel_by_id(config, channel_id)?;
    channel
        .send(&SendMessage::new(message, recipient))
        .await
        .with_context(|| format!("Failed to send message via {channel_id}"))?;
    println!("Message sent via {channel_id}.");
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Health check
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelHealthState {
    Healthy,
    Unhealthy,
    Timeout,
}

fn classify_health_result(
    result: &Result<bool, tokio::time::error::Elapsed>,
) -> ChannelHealthState {
    match result {
        Ok(true) => ChannelHealthState::Healthy,
        Ok(false) => ChannelHealthState::Unhealthy,
        Err(_) => ChannelHealthState::Timeout,
    }
}

pub async fn doctor_channels(config: Config) -> Result<()> {
    #[allow(unused_mut)]
    let mut channels = collect_configured_channels(&config);

    #[cfg(feature = "channel-nostr")]
    if let Some(ref ns) = config.channels_config.nostr {
        channels.push(ConfiguredChannel {
            display_name: "Nostr",
            channel: Arc::new(
                NostrChannel::new(&ns.private_key, ns.relays.clone(), &ns.allowed_pubkeys).await?,
            ),
        });
    }

    if channels.is_empty() {
        println!("No real-time channels configured. Run `zeroclaw onboard` first.");
        return Ok(());
    }

    println!("🩺 ZeroClaw Channel Doctor\n");

    let (mut healthy, mut unhealthy, mut timeout) = (0u32, 0u32, 0u32);

    for configured in channels {
        let result =
            tokio::time::timeout(Duration::from_secs(10), configured.channel.health_check()).await;
        match classify_health_result(&result) {
            ChannelHealthState::Healthy => {
                healthy += 1;
                println!("  ✅ {:<9} healthy", configured.display_name);
            }
            ChannelHealthState::Unhealthy => {
                unhealthy += 1;
                println!(
                    "  ❌ {:<9} unhealthy (auth/config/network)",
                    configured.display_name
                );
            }
            ChannelHealthState::Timeout => {
                timeout += 1;
                println!("  ⏱️  {:<9} timed out (>10s)", configured.display_name);
            }
        }
    }

    if config.channels_config.webhook.is_some() {
        println!("  ℹ️  Webhook   check via `zeroclaw gateway` then GET /health");
    }

    println!("\nSummary: {healthy} healthy, {unhealthy} unhealthy, {timeout} timed out");
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Entry point
// ══════════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_lines)]
pub async fn start_channels(config: Config) -> Result<()> {
    let provider_name = resolved_default_provider(&config);
    let provider_runtime_options = providers::ProviderRuntimeOptions {
        auth_profile_override: None,
        provider_api_url: config.api_url.clone(),
        zeroclaw_dir: config.config_path.parent().map(PathBuf::from),
        secrets_encrypt: config.secrets.encrypt,
        reasoning_enabled: config.runtime.reasoning_enabled,
        provider_timeout_secs: Some(config.provider_timeout_secs),
    };
    let provider: Arc<dyn Provider> = create_and_warm_provider(
        &provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &provider_runtime_options,
    )
    .await?;

    let initial_stamp = config_file_stamp(&config.config_path).await;
    runtime_config_store()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(
            config.config_path.clone(),
            RuntimeConfigState {
                defaults: runtime_defaults_from_config(&config),
                last_applied_stamp: initial_stamp,
            },
        );

    let observer: Arc<dyn Observer> =
        Arc::from(observability::create_observer(&config.observability));
    let runtime: Arc<dyn runtime::RuntimeAdapter> =
        Arc::from(runtime::create_runtime(&config.runtime)?);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));
    let model = resolved_default_model(&config);
    let mem: Arc<dyn Memory> = Arc::from(memory::create_memory_with_storage_and_routes(
        &config.memory,
        &config.embedding_routes,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);

    let (composio_key, composio_entity_id) = if config.composio.enabled {
        (
            config.composio.api_key.as_deref(),
            Some(config.composio.entity_id.as_str()),
        )
    } else {
        (None, None)
    };

    let workspace = config.workspace_dir.clone();
    let tools_registry = Arc::new(tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        Arc::clone(&mem),
        composio_key,
        composio_entity_id,
        &config.browser,
        &config.http_request,
        &config.web_fetch,
        &workspace,
        &config.agents,
        config.api_key.as_deref(),
        &config,
    ));

    let skills = crate::skills::load_skills_with_config(&workspace, &config);

    let native_tools = provider.supports_native_tools();
    // Build tool descriptions for the system prompt, filtering out non-CLI-excluded tools
    let tool_descs = if native_tools {
        Vec::new()
    } else {
        build_tool_descriptions(&config)
    };

    let mut system_prompt = build_system_prompt_inner(
        &workspace,
        &model,
        &tool_descs
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_str()))
            .collect::<Vec<_>>(),
        &skills,
        SystemPromptOptions {
            native_tools,
            skills_prompt_mode: config.skills.prompt_injection_mode,
            bootstrap_max_chars: config.agent.compact_context.then_some(6000),
            identity_config: Some(&config.identity),
        },
    );
    if !native_tools {
        system_prompt.push_str(&build_tool_instructions(tools_registry.as_ref()));
    }

    if !skills.is_empty() {
        println!(
            "  🧩 Skills:   {}",
            skills
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    #[allow(unused_mut)]
    let mut channels: Vec<Arc<dyn Channel>> = collect_configured_channels(&config)
        .into_iter()
        .map(|c| c.channel)
        .collect();

    #[cfg(feature = "channel-nostr")]
    if let Some(ref ns) = config.channels_config.nostr {
        channels.push(Arc::new(
            NostrChannel::new(&ns.private_key, ns.relays.clone(), &ns.allowed_pubkeys).await?,
        ));
    }

    if channels.is_empty() {
        println!("No channels configured. Run `zeroclaw onboard` to set up channels.");
        return Ok(());
    }

    println!("🦀 ZeroClaw Channel Server");
    println!("  🤖 Model:    {model}");
    println!(
        "  🧠 Memory:   {} (auto-save: {})",
        memory::effective_memory_backend_name(
            &config.memory.backend,
            Some(&config.storage.provider.config)
        ),
        if config.memory.auto_save { "on" } else { "off" }
    );
    println!(
        "  📡 Channels: {}",
        channels
            .iter()
            .map(|c| c.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("\n  Listening for messages... (Ctrl+C to stop)\n");

    crate::health::mark_component_ok("channels");

    let initial_backoff_secs = config
        .reliability
        .channel_initial_backoff_secs
        .max(DEFAULT_CHANNEL_INITIAL_BACKOFF_SECS);
    let max_backoff_secs = config
        .reliability
        .channel_max_backoff_secs
        .max(DEFAULT_CHANNEL_MAX_BACKOFF_SECS);

    let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(100);
    let mut handles = Vec::new();
    for ch in &channels {
        handles.push(spawn_supervised_listener(
            ch.clone(),
            tx.clone(),
            initial_backoff_secs,
            max_backoff_secs,
        ));
    }
    drop(tx);

    let channels_by_name: Arc<HashMap<String, Arc<dyn Channel>>> = Arc::new(
        channels
            .iter()
            .map(|ch| (ch.name().to_string(), Arc::clone(ch)))
            .collect(),
    );
    let max_in_flight_messages = compute_max_in_flight_messages(channels.len());
    println!("  🚦 In-flight message limit: {max_in_flight_messages}");

    let message_timeout_secs =
        effective_channel_message_timeout_secs(config.channels_config.message_timeout_secs);
    let interrupt_on_new_message = config
        .channels_config
        .telegram
        .as_ref()
        .is_some_and(|tg| tg.interrupt_on_new_message);

    let mut provider_cache_seed: HashMap<String, Arc<dyn Provider>> = HashMap::new();
    provider_cache_seed.insert(provider_name.clone(), Arc::clone(&provider));

    let runtime_ctx = Arc::new(ChannelRuntimeContext {
        channels_by_name,
        provider: Arc::clone(&provider),
        default_provider: Arc::new(provider_name),
        memory: Arc::clone(&mem),
        tools_registry: Arc::clone(&tools_registry),
        observer,
        system_prompt: Arc::new(system_prompt),
        model: Arc::new(model),
        temperature: config.default_temperature,
        auto_save_memory: config.memory.auto_save,
        max_tool_iterations: config.agent.max_tool_iterations,
        min_relevance_score: config.memory.min_relevance_score,
        conversation_histories: Arc::new(Mutex::new(HashMap::new())),
        provider_cache: Arc::new(Mutex::new(provider_cache_seed)),
        route_overrides: Arc::new(Mutex::new(HashMap::new())),
        api_key: config.api_key.clone(),
        api_url: config.api_url.clone(),
        reliability: Arc::new(config.reliability.clone()),
        provider_runtime_options,
        workspace_dir: Arc::new(config.workspace_dir.clone()),
        message_timeout_secs,
        interrupt_on_new_message,
        multimodal: config.multimodal.clone(),
        hooks: build_hook_runner(&config),
        non_cli_excluded_tools: Arc::new(config.autonomy.non_cli_excluded_tools.clone()),
        tool_call_dedup_exempt: Arc::new(config.agent.tool_call_dedup_exempt.clone()),
        model_routes: Arc::new(config.model_routes.clone()),
    });

    run_message_dispatch_loop(rx, runtime_ctx, max_in_flight_messages).await;
    for h in handles {
        let _ = h.await;
    }
    Ok(())
}

/// Build the tool description list for the system prompt, excluding non-CLI tools.
fn build_tool_descriptions(config: &Config) -> Vec<(String, String)> {
    let mut descs: Vec<(String, String)> = vec![
        ("shell".into(), "Execute terminal commands. Use when: running local checks, build/test commands, diagnostics. Don't use when: a safer dedicated tool exists, or command is destructive without approval.".into()),
        ("file_read".into(), "Read file contents. Use when: inspecting project files, configs, logs. Don't use when: a targeted search is enough.".into()),
        ("file_write".into(), "Write file contents. Use when: applying focused edits, scaffolding files, updating docs/code. Don't use when: side effects are unclear or file ownership is uncertain.".into()),
        ("memory_store".into(), "Save to memory. Use when: preserving durable preferences, decisions, key context. Don't use when: information is transient/noisy/sensitive without need.".into()),
        ("memory_recall".into(), "Search memory. Use when: retrieving prior decisions, user preferences, historical context. Don't use when: answer is already in current context.".into()),
        ("memory_forget".into(), "Delete a memory entry. Use when: memory is incorrect/stale or explicitly requested for removal. Don't use when: impact is uncertain.".into()),
    ];

    if config.browser.enabled {
        descs.push((
            "browser_open".into(),
            "Open approved HTTPS URLs in system browser (allowlist-only, no scraping)".into(),
        ));
    }
    if config.composio.enabled {
        descs.push(("composio".into(), "Execute actions on 1000+ apps via Composio (Gmail, Notion, GitHub, Slack, etc.). Use action='list' to discover actions, 'list_accounts' to retrieve connected account IDs, 'execute' to run (optionally with connected_account_id), and 'connect' for OAuth.".into()));
    }
    descs.push(("schedule".into(), "Manage scheduled tasks (create/list/get/cancel/pause/resume). Supports recurring cron and one-shot delays.".into()));
    descs.push(("pushover".into(), "Send a Pushover notification to your device. Requires PUSHOVER_TOKEN and PUSHOVER_USER_KEY in .env file.".into()));
    if !config.agents.is_empty() {
        descs.push(("delegate".into(), "Delegate a subtask to a specialized agent. Use when: a task benefits from a different model (e.g. fast summarization, deep reasoning, code generation). The sub-agent runs a single prompt and returns its response.".into()));
    }

    let excluded = &config.autonomy.non_cli_excluded_tools;
    if !excluded.is_empty() {
        descs.retain(|(name, _)| !excluded.iter().any(|ex| ex == name));
    }

    descs
}

/// Build the hook runner from config, or return `None` if hooks are disabled.
fn build_hook_runner(config: &Config) -> Option<Arc<crate::hooks::HookRunner>> {
    if !config.hooks.enabled {
        return None;
    }

    let mut runner = crate::hooks::HookRunner::new();
    if config.hooks.builtin.command_logger {
        runner.register(Box::new(crate::hooks::builtin::CommandLoggerHook::new()));
    }
    if config.hooks.builtin.webhook_audit.enabled {
        runner.register(Box::new(crate::hooks::builtin::WebhookAuditHook::new(
            config.hooks.builtin.webhook_audit.clone(),
        )));
    }
    Some(Arc::new(runner))
}
