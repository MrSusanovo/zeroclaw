use super::traits::{Channel, ChannelMessage, SendMessage};
use super::{
    append_sender_turn, compact_sender_history, extract_tool_context_summary,
    is_context_window_overflow_error, rollback_orphan_user_turn, sanitize_channel_response,
    truncate_with_ellipsis, ChannelRouteSelection, ChannelRuntimeContext,
    CHANNEL_HOOK_MAX_OUTBOUND_CHARS,
};
use crate::agent::loop_::scrub_credentials;
use crate::observability::runtime_trace;
use crate::providers;
use crate::providers::ChatMessage;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Send a message via draft finalization if a draft exists, otherwise send fresh.
///
/// When `draft_id` is provided and finalization fails, logs a warning and falls
/// back to sending a new message so the reply is never silently dropped.
async fn deliver_or_finalize(
    channel: &Arc<dyn Channel>,
    reply_target: &str,
    thread_ts: Option<String>,
    draft_id: Option<&str>,
    text: impl Into<String>,
) {
    let text = text.into();
    if let Some(draft_id) = draft_id {
        if let Err(e) = channel.finalize_draft(reply_target, draft_id, &text).await {
            tracing::warn!("Failed to finalize draft: {e}; sending as new message");
            let _ = channel
                .send(&SendMessage::new(text, reply_target).in_thread(thread_ts))
                .await;
        }
    } else if let Err(e) = channel
        .send(&SendMessage::new(text, reply_target).in_thread(thread_ts))
        .await
    {
        eprintln!("  ❌ Failed to reply on {}: {e}", channel.name());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceEventType {
    Cancelled,
    Error,
    Timeout,
    Outbound,
}

fn trace_channel_event(
    msg: &ChannelMessage,
    route: &ChannelRouteSelection,
    event_type: TraceEventType,
    reason: Option<&str>,
    elapsed_ms: u128,
    extra: Option<serde_json::Value>,
) {
    let (event_name, success) = match event_type {
        TraceEventType::Cancelled => ("channel_message_cancelled", false),
        TraceEventType::Error => ("channel_message_error", false),
        TraceEventType::Timeout => ("channel_message_timeout", false),
        TraceEventType::Outbound => ("channel_message_outbound", true),
    };

    let mut payload = serde_json::json!({
        "sender": msg.sender,
        "elapsed_ms": elapsed_ms,
    });
    if let (Some(base), Some(extra_map)) = (
        payload.as_object_mut(),
        extra.as_ref().and_then(|e| e.as_object()),
    ) {
        base.extend(extra_map.clone());
    }

    runtime_trace::record_event(
        event_name,
        Some(msg.channel.as_str()),
        Some(route.provider.as_str()),
        Some(route.model.as_str()),
        None,
        Some(success),
        reason,
        payload,
    );
}

// ── Handlers ─────────────────────────────────────────────────────────────────

pub(super) async fn handle_channel_message_timeout(
    ctx: &ChannelRuntimeContext,
    history_key: &str,
    msg: &ChannelMessage,
    route: ChannelRouteSelection,
    target_channel: Option<&Arc<dyn Channel>>,
    draft_message_id: Option<&str>,
    started_at: Instant,
    timeout_budget_secs: u64,
) {
    let elapsed_ms = started_at.elapsed().as_millis();
    let timeout_msg = format!(
        "LLM response timed out after {}s (base={}s, max_tool_iterations={})",
        timeout_budget_secs, ctx.message_timeout_secs, ctx.max_tool_iterations
    );

    trace_channel_event(
        &msg,
        &route,
        TraceEventType::Timeout,
        Some(&timeout_msg),
        elapsed_ms,
        None,
    );
    eprintln!("  ❌ {} (elapsed: {}ms)", timeout_msg, elapsed_ms);

    append_sender_turn(
        ctx,
        history_key,
        ChatMessage::assistant("[Task timed out — not continuing this request]"),
    );

    if let Some(channel) = target_channel {
        deliver_or_finalize(
            channel,
            &msg.reply_target,
            msg.thread_ts.clone(),
            draft_message_id,
            "⚠️ Request timed out while waiting for the model. Please try again.",
        )
        .await;
    }
}

pub(super) async fn handle_cancelled_message(
    msg: &ChannelMessage,
    route: ChannelRouteSelection,
    target_channel: Option<&Arc<dyn Channel>>,
    draft_message_id: Option<&str>,
    started_at: Instant,
) {
    tracing::info!(
        channel = %msg.channel,
        sender = %msg.sender,
        "Cancelled in-flight channel request due to newer message"
    );
    trace_channel_event(
        &msg,
        &route,
        TraceEventType::Cancelled,
        Some("cancelled due to newer inbound message"),
        started_at.elapsed().as_millis(),
        None,
    );

    if let (Some(channel), Some(draft_id)) = (target_channel, draft_message_id) {
        if let Err(err) = channel.cancel_draft(&msg.reply_target, draft_id).await {
            tracing::debug!("Failed to cancel draft on {}: {err}", channel.name());
        }
    }
}

pub(super) async fn handle_llm_error(
    err: anyhow::Error,
    ctx: &ChannelRuntimeContext,
    msg: &ChannelMessage,
    route: ChannelRouteSelection,
    target_channel: Option<&Arc<dyn Channel>>,
    draft_message_id: Option<&str>,
    started_at: Instant,
    history_key: &str,
    cancellation_token: &CancellationToken,
) {
    let elapsed_ms = started_at.elapsed().as_millis();

    if crate::agent::loop_::is_tool_loop_cancelled(&err) || cancellation_token.is_cancelled() {
        tracing::info!(
            channel = %msg.channel,
            sender = %msg.sender,
            "Cancelled in-flight channel request due to newer message"
        );
        trace_channel_event(
            &msg,
            &route,
            TraceEventType::Cancelled,
            Some("cancelled during tool-call loop"),
            elapsed_ms,
            None,
        );
        if let (Some(channel), Some(draft_id)) = (target_channel, draft_message_id) {
            if let Err(e) = channel.cancel_draft(&msg.reply_target, draft_id).await {
                tracing::debug!("Failed to cancel draft on {}: {e}", channel.name());
            }
        }
        return;
    }

    if is_context_window_overflow_error(&err) {
        let compacted = compact_sender_history(ctx, history_key);
        eprintln!(
            "  ⚠️ Context window exceeded after {}ms; sender history compacted={}",
            elapsed_ms, compacted
        );
        trace_channel_event(
            &msg,
            &route,
            TraceEventType::Error,
            Some("context window exceeded"),
            elapsed_ms,
            Some(serde_json::json!({ "history_compacted": compacted })),
        );
        let error_text = if compacted {
            "⚠️ Context window exceeded for this conversation. I compacted recent history and kept the latest context. Please resend your last message."
        } else {
            "⚠️ Context window exceeded for this conversation. Please resend your last message."
        };
        if let Some(channel) = target_channel {
            deliver_or_finalize(
                channel,
                &msg.reply_target,
                msg.thread_ts.clone(),
                draft_message_id,
                error_text,
            )
            .await;
        }
        return;
    }

    // General LLM error
    eprintln!("  ❌ LLM error after {}ms: {err}", elapsed_ms);
    let safe_error = providers::sanitize_api_error(&err.to_string());
    trace_channel_event(
        &msg,
        &route,
        TraceEventType::Error,
        Some(&safe_error),
        elapsed_ms,
        None,
    );

    let should_rollback = err
        .downcast_ref::<providers::ProviderCapabilityError>()
        .is_some_and(|c| c.capability.eq_ignore_ascii_case("vision"));
    let rolled_back = should_rollback && rollback_orphan_user_turn(ctx, history_key, &msg.content);

    if !rolled_back {
        append_sender_turn(
            ctx,
            history_key,
            ChatMessage::assistant("[Task failed — not continuing this request]"),
        );
    }
    if let Some(channel) = target_channel {
        deliver_or_finalize(
            channel,
            &msg.reply_target,
            msg.thread_ts.clone(),
            draft_message_id,
            format!("⚠️ Error: {err}"),
        )
        .await;
    }
}

pub(super) async fn handle_llm_ok(
    ctx: &ChannelRuntimeContext,
    msg: &mut ChannelMessage,
    route: &ChannelRouteSelection,
    target_channel: Option<&Arc<dyn Channel>>,
    draft_message_id: Option<&str>,
    started_at: Instant,
    history: &[ChatMessage],
    history_key: &str,
    history_len_before_tools: usize,
    response: String,
) {
    let elapsed_ms = started_at.elapsed().as_millis();
    let mut outbound_response = response;

    // ── Hook: on_message_sending ──────────────────────────────────────────────
    if let Some(hooks) = &ctx.hooks {
        match hooks
            .run_on_message_sending(
                msg.channel.clone(),
                msg.reply_target.clone(),
                outbound_response.clone(),
            )
            .await
        {
            crate::hooks::HookResult::Cancel(reason) => {
                tracing::info!(%reason, "outgoing message suppressed by hook");
                return;
            }
            crate::hooks::HookResult::Continue((hook_channel, hook_recipient, mut modified)) => {
                if hook_channel != msg.channel || hook_recipient != msg.reply_target {
                    tracing::warn!(
                        from_channel = %msg.channel, from_recipient = %msg.reply_target,
                        to_channel = %hook_channel, to_recipient = %hook_recipient,
                        "on_message_sending attempted to rewrite channel routing; only content mutation is applied"
                    );
                }
                if modified.chars().count() > CHANNEL_HOOK_MAX_OUTBOUND_CHARS {
                    tracing::warn!(
                        limit = CHANNEL_HOOK_MAX_OUTBOUND_CHARS,
                        attempted = modified.chars().count(),
                        "hook-modified outbound content exceeded limit; truncating"
                    );
                    modified = truncate_with_ellipsis(&modified, CHANNEL_HOOK_MAX_OUTBOUND_CHARS);
                }
                if modified != outbound_response {
                    tracing::info!(
                        channel = %msg.channel, sender = %msg.sender,
                        before_len = outbound_response.chars().count(),
                        after_len = modified.chars().count(),
                        "outgoing message content modified by hook"
                    );
                }
                outbound_response = modified;
            }
        }
    }

    // ── Sanitize & deliver ────────────────────────────────────────────────────
    let sanitized = sanitize_channel_response(&outbound_response, ctx.tools_registry.as_ref());
    let delivered = if sanitized.is_empty() && !outbound_response.trim().is_empty() {
        "I encountered malformed tool-call output and could not produce a safe reply. Please try again.".to_string()
    } else {
        sanitized
    };

    trace_channel_event(
        msg,
        route,
        TraceEventType::Outbound,
        None,
        elapsed_ms,
        Some(serde_json::json!({ "response": scrub_credentials(&delivered) })),
    );

    let tool_summary = extract_tool_context_summary(history, history_len_before_tools);
    let history_response = if tool_summary.is_empty() || msg.channel == "telegram" {
        delivered.clone()
    } else {
        format!("{tool_summary}\n{delivered}")
    };

    append_sender_turn(ctx, history_key, ChatMessage::assistant(&history_response));
    println!(
        "  🤖 Reply ({}ms): {}",
        elapsed_ms,
        truncate_with_ellipsis(&delivered, 80)
    );

    if let Some(channel) = target_channel {
        deliver_or_finalize(
            channel,
            &msg.reply_target,
            msg.thread_ts.clone(),
            draft_message_id,
            delivered,
        )
        .await;
    }
}
