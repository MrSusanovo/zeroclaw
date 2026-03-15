
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Memory, MemoryCategory, SqliteMemory};
    use crate::observability::NoopObserver;
    use crate::providers::{ChatMessage, Provider};
    use crate::tools::{Tool, ToolResult};
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_workspace() -> TempDir {
        let tmp = TempDir::new().unwrap();
        // Create minimal workspace files
        std::fs::write(tmp.path().join("SOUL.md"), "# Soul\nBe helpful.").unwrap();
        std::fs::write(tmp.path().join("IDENTITY.md"), "# Identity\nName: ZeroClaw").unwrap();
        std::fs::write(tmp.path().join("USER.md"), "# User\nName: Test User").unwrap();
        std::fs::write(
            tmp.path().join("AGENTS.md"),
            "# Agents\nFollow instructions.",
        )
        .unwrap();
        std::fs::write(tmp.path().join("TOOLS.md"), "# Tools\nUse shell carefully.").unwrap();
        std::fs::write(
            tmp.path().join("HEARTBEAT.md"),
            "# Heartbeat\nCheck status.",
        )
        .unwrap();
        std::fs::write(tmp.path().join("MEMORY.md"), "# Memory\nUser likes Rust.").unwrap();
        tmp
    }

    #[test]
    fn effective_channel_message_timeout_secs_clamps_to_minimum() {
        assert_eq!(
            effective_channel_message_timeout_secs(0),
            MIN_CHANNEL_MESSAGE_TIMEOUT_SECS
        );
        assert_eq!(
            effective_channel_message_timeout_secs(15),
            MIN_CHANNEL_MESSAGE_TIMEOUT_SECS
        );
        assert_eq!(effective_channel_message_timeout_secs(300), 300);
    }

    #[test]
    fn channel_message_timeout_budget_scales_with_tool_iterations() {
        assert_eq!(channel_message_timeout_budget_secs(300, 1), 300);
        assert_eq!(channel_message_timeout_budget_secs(300, 2), 600);
        assert_eq!(channel_message_timeout_budget_secs(300, 3), 900);
    }

    #[test]
    fn channel_message_timeout_budget_uses_safe_defaults_and_cap() {
        // 0 iterations falls back to 1x timeout budget.
        assert_eq!(channel_message_timeout_budget_secs(300, 0), 300);
        // Large iteration counts are capped to avoid runaway waits.
        assert_eq!(
            channel_message_timeout_budget_secs(300, 10),
            300 * CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP
        );
    }

    #[test]
    fn context_window_overflow_error_detector_matches_known_messages() {
        let overflow_err = anyhow::anyhow!(
            "OpenAI Codex stream error: Your input exceeds the context window of this model."
        );
        assert!(is_context_window_overflow_error(&overflow_err));

        let other_err =
            anyhow::anyhow!("OpenAI Codex API error (502 Bad Gateway): error code: 502");
        assert!(!is_context_window_overflow_error(&other_err));
    }

    #[test]
    fn memory_context_skip_rules_exclude_history_blobs() {
        assert!(should_skip_memory_context_entry(
            "telegram_123_history",
            r#"[{"role":"user"}]"#
        ));
        assert!(should_skip_memory_context_entry(
            "assistant_resp_legacy",
            "fabricated memory"
        ));
        assert!(!should_skip_memory_context_entry("telegram_123_45", "hi"));

        // Entries containing image markers must be skipped to prevent
        // auto-saved photo messages from duplicating image blocks (#2403).
        assert!(should_skip_memory_context_entry(
            "telegram_user_msg_99",
            "[IMAGE:/tmp/workspace/photo_1_2.jpg]"
        ));
        assert!(should_skip_memory_context_entry(
            "telegram_user_msg_100",
            "[IMAGE:/tmp/workspace/photo_1_2.jpg]\n\nCheck this screenshot"
        ));
        // Plain text without image markers should not be skipped.
        assert!(!should_skip_memory_context_entry(
            "telegram_user_msg_101",
            "Please describe the image"
        ));
    }

    #[test]
    fn normalize_cached_channel_turns_merges_consecutive_user_turns() {
        let turns = vec![
            ChatMessage::user("forwarded content"),
            ChatMessage::user("summarize this"),
        ];

        let normalized = normalize_cached_channel_turns(turns);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].role, "user");
        assert!(normalized[0].content.contains("forwarded content"));
        assert!(normalized[0].content.contains("summarize this"));
    }

    #[test]
    fn normalize_cached_channel_turns_merges_consecutive_assistant_turns() {
        let turns = vec![
            ChatMessage::user("first user"),
            ChatMessage::assistant("assistant part 1"),
            ChatMessage::assistant("assistant part 2"),
            ChatMessage::user("next user"),
        ];

        let normalized = normalize_cached_channel_turns(turns);
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0].role, "user");
        assert_eq!(normalized[1].role, "assistant");
        assert_eq!(normalized[2].role, "user");
        assert!(normalized[1].content.contains("assistant part 1"));
        assert!(normalized[1].content.contains("assistant part 2"));
    }

    /// Verify that an orphan user turn followed by a failure-marker assistant
    /// turn normalizes correctly, so the LLM sees the failed request as closed
    /// and does not re-execute it on the next user message.
    #[test]
    fn normalize_preserves_failure_marker_after_orphan_user_turn() {
        let turns = vec![
            ChatMessage::user("download something from GitHub"),
            ChatMessage::assistant("[Task failed — not continuing this request]"),
            ChatMessage::user("what is WAL?"),
        ];

        let normalized = normalize_cached_channel_turns(turns);
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0].role, "user");
        assert_eq!(normalized[1].role, "assistant");
        assert!(normalized[1].content.contains("Task failed"));
        assert_eq!(normalized[2].role, "user");
        assert_eq!(normalized[2].content, "what is WAL?");
    }

    /// Same as above but for the timeout variant.
    #[test]
    fn normalize_preserves_timeout_marker_after_orphan_user_turn() {
        let turns = vec![
            ChatMessage::user("run a long task"),
            ChatMessage::assistant("[Task timed out — not continuing this request]"),
            ChatMessage::user("next question"),
        ];

        let normalized = normalize_cached_channel_turns(turns);
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[1].role, "assistant");
        assert!(normalized[1].content.contains("Task timed out"));
        assert_eq!(normalized[2].content, "next question");
    }

    #[test]
    fn compact_sender_history_keeps_recent_truncated_messages() {
        let mut histories = HashMap::new();
        let sender = "telegram_u1".to_string();
        histories.insert(
            sender.clone(),
            (0..20)
                .map(|idx| {
                    let content = format!("msg-{idx}-{}", "x".repeat(700));
                    if idx % 2 == 0 {
                        ChatMessage::user(content)
                    } else {
                        ChatMessage::assistant(content)
                    }
                })
                .collect::<Vec<_>>(),
        );

        let ctx = ChannelRuntimeContext {
            channels_by_name: Arc::new(HashMap::new()),
            provider: Arc::new(DummyProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("system".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(histories)),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        };

        assert!(compact_sender_history(&ctx, &sender));

        let locked_histories = ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let kept = locked_histories
            .get(&sender)
            .expect("sender history should remain");
        assert_eq!(kept.len(), CHANNEL_HISTORY_COMPACT_KEEP_MESSAGES);
        assert!(kept.iter().all(|turn| {
            let len = turn.content.chars().count();
            len <= CHANNEL_HISTORY_COMPACT_CONTENT_CHARS
                || (len <= CHANNEL_HISTORY_COMPACT_CONTENT_CHARS + 3
                    && turn.content.ends_with("..."))
        }));
    }

    #[test]
    fn append_sender_turn_stores_single_turn_per_call() {
        let sender = "telegram_u2".to_string();
        let ctx = ChannelRuntimeContext {
            channels_by_name: Arc::new(HashMap::new()),
            provider: Arc::new(DummyProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("system".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        };

        append_sender_turn(&ctx, &sender, ChatMessage::user("hello"));

        let histories = ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let turns = histories.get(&sender).expect("sender history should exist");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content, "hello");
    }

    #[test]
    fn rollback_orphan_user_turn_removes_only_latest_matching_user_turn() {
        let sender = "telegram_u3".to_string();
        let mut histories = HashMap::new();
        histories.insert(
            sender.clone(),
            vec![
                ChatMessage::user("first"),
                ChatMessage::assistant("ok"),
                ChatMessage::user("pending"),
            ],
        );
        let ctx = ChannelRuntimeContext {
            channels_by_name: Arc::new(HashMap::new()),
            provider: Arc::new(DummyProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("system".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(histories)),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        };

        assert!(rollback_orphan_user_turn(&ctx, &sender, "pending"));

        let locked_histories = ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let turns = locked_histories
            .get(&sender)
            .expect("sender history should remain");
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].content, "first");
        assert_eq!(turns[1].content, "ok");
    }

    struct DummyProvider;

    #[async_trait::async_trait]
    impl Provider for DummyProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    #[derive(Default)]
    struct RecordingChannel {
        sent_messages: tokio::sync::Mutex<Vec<String>>,
        start_typing_calls: AtomicUsize,
        stop_typing_calls: AtomicUsize,
        reactions_added: tokio::sync::Mutex<Vec<(String, String, String)>>,
        reactions_removed: tokio::sync::Mutex<Vec<(String, String, String)>>,
    }

    #[derive(Default)]
    struct TelegramRecordingChannel {
        sent_messages: tokio::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl Channel for TelegramRecordingChannel {
        fn name(&self) -> &str {
            "telegram"
        }

        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.sent_messages
                .lock()
                .await
                .push(format!("{}:{}", message.recipient, message.content));
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
            Ok(())
        }

        async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl Channel for RecordingChannel {
        fn name(&self) -> &str {
            "test-channel"
        }

        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.sent_messages
                .lock()
                .await
                .push(format!("{}:{}", message.recipient, message.content));
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
            self.start_typing_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
            self.stop_typing_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn add_reaction(
            &self,
            channel_id: &str,
            message_id: &str,
            emoji: &str,
        ) -> anyhow::Result<()> {
            self.reactions_added.lock().await.push((
                channel_id.to_string(),
                message_id.to_string(),
                emoji.to_string(),
            ));
            Ok(())
        }

        async fn remove_reaction(
            &self,
            channel_id: &str,
            message_id: &str,
            emoji: &str,
        ) -> anyhow::Result<()> {
            self.reactions_removed.lock().await.push((
                channel_id.to_string(),
                message_id.to_string(),
                emoji.to_string(),
            ));
            Ok(())
        }
    }

    struct SlowProvider {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl Provider for SlowProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            tokio::time::sleep(self.delay).await;
            Ok(format!("echo: {message}"))
        }
    }

    struct ToolCallingProvider;

    fn tool_call_payload() -> String {
        r#"<tool_call>
{"name":"mock_price","arguments":{"symbol":"BTC"}}
</tool_call>"#
            .to_string()
    }

    fn tool_call_payload_with_alias_tag() -> String {
        r#"<toolcall>
{"name":"mock_price","arguments":{"symbol":"BTC"}}
</toolcall>"#
            .to_string()
    }

    #[async_trait::async_trait]
    impl Provider for ToolCallingProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok(tool_call_payload())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            let has_tool_results = messages
                .iter()
                .any(|msg| msg.role == "user" && msg.content.contains("[Tool results]"));
            if has_tool_results {
                Ok("BTC is currently around $65,000 based on latest tool output.".to_string())
            } else {
                Ok(tool_call_payload())
            }
        }
    }

    struct ToolCallingAliasProvider;

    #[async_trait::async_trait]
    impl Provider for ToolCallingAliasProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok(tool_call_payload_with_alias_tag())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            let has_tool_results = messages
                .iter()
                .any(|msg| msg.role == "user" && msg.content.contains("[Tool results]"));
            if has_tool_results {
                Ok("BTC alias-tag flow resolved to final text output.".to_string())
            } else {
                Ok(tool_call_payload_with_alias_tag())
            }
        }
    }

    struct RawToolArtifactProvider;

    #[async_trait::async_trait]
    impl Provider for RawToolArtifactProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("fallback".to_string())
        }

        async fn chat_with_history(
            &self,
            _messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok(r#"{"name":"mock_price","parameters":{"symbol":"BTC"}}
{"result":{"symbol":"BTC","price_usd":65000}}
BTC is currently around $65,000 based on latest tool output."#
                .to_string())
        }
    }

    struct IterativeToolProvider {
        required_tool_iterations: usize,
    }

    impl IterativeToolProvider {
        fn completed_tool_iterations(messages: &[ChatMessage]) -> usize {
            messages
                .iter()
                .filter(|msg| msg.role == "user" && msg.content.contains("[Tool results]"))
                .count()
        }
    }

    #[async_trait::async_trait]
    impl Provider for IterativeToolProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok(tool_call_payload())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            let completed_iterations = Self::completed_tool_iterations(messages);
            if completed_iterations >= self.required_tool_iterations {
                Ok(format!(
                    "Completed after {completed_iterations} tool iterations."
                ))
            } else {
                Ok(tool_call_payload())
            }
        }
    }

    #[derive(Default)]
    struct HistoryCaptureProvider {
        calls: std::sync::Mutex<Vec<Vec<(String, String)>>>,
    }

    #[async_trait::async_trait]
    impl Provider for HistoryCaptureProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("fallback".to_string())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            let snapshot = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>();
            let mut calls = self.calls.lock().unwrap_or_else(|e| e.into_inner());
            calls.push(snapshot);
            Ok(format!("response-{}", calls.len()))
        }
    }

    struct DelayedHistoryCaptureProvider {
        delay: Duration,
        calls: std::sync::Mutex<Vec<Vec<(String, String)>>>,
    }

    #[async_trait::async_trait]
    impl Provider for DelayedHistoryCaptureProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("fallback".to_string())
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            let snapshot = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>();
            let call_index = {
                let mut calls = self.calls.lock().unwrap_or_else(|e| e.into_inner());
                calls.push(snapshot);
                calls.len()
            };
            tokio::time::sleep(self.delay).await;
            Ok(format!("response-{call_index}"))
        }
    }

    struct MockPriceTool;

    #[derive(Default)]
    struct ModelCaptureProvider {
        call_count: AtomicUsize,
        models: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl Provider for ModelCaptureProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("fallback".to_string())
        }

        async fn chat_with_history(
            &self,
            _messages: &[ChatMessage],
            model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.models
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(model.to_string());
            Ok("ok".to_string())
        }
    }

    #[async_trait::async_trait]
    impl Tool for MockPriceTool {
        fn name(&self) -> &str {
            "mock_price"
        }

        fn description(&self) -> &str {
            "Return a mocked BTC price"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": { "type": "string" }
                },
                "required": ["symbol"]
            })
        }

        async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
            let symbol = args.get("symbol").and_then(serde_json::Value::as_str);
            if symbol != Some("BTC") {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("unexpected symbol".to_string()),
                });
            }

            Ok(ToolResult {
                success: true,
                output: r#"{"symbol":"BTC","price_usd":65000}"#.to_string(),
                error: None,
            })
        }
    }

    #[tokio::test]
    async fn process_channel_message_executes_tool_calls_instead_of_sending_raw_json() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(ToolCallingProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-42".to_string(),
                content: "What is the BTC price now?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert!(!sent_messages.is_empty());
        let reply = sent_messages.last().unwrap();
        assert!(reply.starts_with("chat-42:"));
        assert!(reply.contains("BTC is currently around"));
        assert!(!reply.contains("\"tool_calls\""));
        assert!(!reply.contains("mock_price"));
    }

    #[tokio::test]
    async fn process_channel_message_telegram_does_not_persist_tool_summary_prefix() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(ToolCallingProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx.clone(),
            traits::ChannelMessage {
                id: "msg-telegram-tool-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-telegram".to_string(),
                content: "What is the BTC price now?".to_string(),
                channel: "telegram".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert!(!sent_messages.is_empty());
        let reply = sent_messages.last().unwrap();
        assert!(reply.contains("BTC is currently around"));

        let histories = runtime_ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let turns = histories
            .get("telegram_alice")
            .expect("telegram history should be stored");
        let assistant_turn = turns
            .iter()
            .rev()
            .find(|turn| turn.role == "assistant")
            .expect("assistant turn should be present");
        assert!(
            !assistant_turn.content.contains("[Used tools:"),
            "telegram history should not persist tool-summary prefix"
        );
    }

    #[tokio::test]
    async fn process_channel_message_strips_unexecuted_tool_json_artifacts_from_reply() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(RawToolArtifactProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-raw-json".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-raw".to_string(),
                content: "What is the BTC price now?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 3,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert_eq!(sent_messages.len(), 1);
        assert!(sent_messages[0].starts_with("chat-raw:"));
        assert!(sent_messages[0].contains("BTC is currently around"));
        assert!(!sent_messages[0].contains("\"name\":\"mock_price\""));
        assert!(!sent_messages[0].contains("\"result\""));
    }

    #[tokio::test]
    async fn process_channel_message_executes_tool_calls_with_alias_tags() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(ToolCallingAliasProvider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-2".to_string(),
                sender: "bob".to_string(),
                reply_target: "chat-84".to_string(),
                content: "What is the BTC price now?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 2,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert!(!sent_messages.is_empty());
        let reply = sent_messages.last().unwrap();
        assert!(reply.starts_with("chat-84:"));
        assert!(reply.contains("alias-tag flow resolved"));
        assert!(!reply.contains("<toolcall>"));
        assert!(!reply.contains("mock_price"));
    }

    #[tokio::test]
    async fn process_channel_message_handles_models_command_without_llm_call() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let default_provider_impl = Arc::new(ModelCaptureProvider::default());
        let default_provider: Arc<dyn Provider> = default_provider_impl.clone();
        let fallback_provider_impl = Arc::new(ModelCaptureProvider::default());
        let fallback_provider: Arc<dyn Provider> = fallback_provider_impl.clone();

        let mut provider_cache_seed: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        provider_cache_seed.insert("test-provider".to_string(), Arc::clone(&default_provider));
        provider_cache_seed.insert("openrouter".to_string(), fallback_provider);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::clone(&default_provider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("default-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(provider_cache_seed)),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx.clone(),
            traits::ChannelMessage {
                id: "msg-cmd-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "/models openrouter".to_string(),
                channel: "telegram".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent = channel_impl.sent_messages.lock().await;
        assert_eq!(sent.len(), 1);
        assert!(sent[0].contains("Provider switched to `openrouter`"));

        let route_key = "telegram_alice";
        let route = runtime_ctx
            .route_overrides
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(route_key)
            .cloned()
            .expect("route should be stored for sender");
        assert_eq!(route.provider, "openrouter");
        assert_eq!(route.model, "default-model");

        assert_eq!(default_provider_impl.call_count.load(Ordering::SeqCst), 0);
        assert_eq!(fallback_provider_impl.call_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn process_channel_message_uses_route_override_provider_and_model() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let default_provider_impl = Arc::new(ModelCaptureProvider::default());
        let default_provider: Arc<dyn Provider> = default_provider_impl.clone();
        let routed_provider_impl = Arc::new(ModelCaptureProvider::default());
        let routed_provider: Arc<dyn Provider> = routed_provider_impl.clone();

        let mut provider_cache_seed: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        provider_cache_seed.insert("test-provider".to_string(), Arc::clone(&default_provider));
        provider_cache_seed.insert("openrouter".to_string(), routed_provider);

        let route_key = "telegram_alice".to_string();
        let mut route_overrides = HashMap::new();
        route_overrides.insert(
            route_key,
            ChannelRouteSelection {
                provider: "openrouter".to_string(),
                model: "route-model".to_string(),
            },
        );

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::clone(&default_provider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("default-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(provider_cache_seed)),
            route_overrides: Arc::new(Mutex::new(route_overrides)),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-routed-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "hello routed provider".to_string(),
                channel: "telegram".to_string(),
                timestamp: 2,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        assert_eq!(default_provider_impl.call_count.load(Ordering::SeqCst), 0);
        assert_eq!(routed_provider_impl.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            routed_provider_impl
                .models
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_slice(),
            &["route-model".to_string()]
        );
    }

    #[tokio::test]
    async fn process_channel_message_prefers_cached_default_provider_instance() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let startup_provider_impl = Arc::new(ModelCaptureProvider::default());
        let startup_provider: Arc<dyn Provider> = startup_provider_impl.clone();
        let reloaded_provider_impl = Arc::new(ModelCaptureProvider::default());
        let reloaded_provider: Arc<dyn Provider> = reloaded_provider_impl.clone();

        let mut provider_cache_seed: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        provider_cache_seed.insert("test-provider".to_string(), reloaded_provider);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::clone(&startup_provider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("default-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(provider_cache_seed)),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-default-provider-cache".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "hello cached default provider".to_string(),
                channel: "telegram".to_string(),
                timestamp: 3,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        assert_eq!(startup_provider_impl.call_count.load(Ordering::SeqCst), 0);
        assert_eq!(reloaded_provider_impl.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn process_channel_message_uses_runtime_default_model_from_store() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let provider_impl = Arc::new(ModelCaptureProvider::default());
        let provider: Arc<dyn Provider> = provider_impl.clone();
        let mut provider_cache_seed: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        provider_cache_seed.insert("test-provider".to_string(), Arc::clone(&provider));

        let temp = tempfile::TempDir::new().expect("temp dir");
        let config_path = temp.path().join("config.toml");

        {
            let mut store = runtime_config_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            store.insert(
                config_path.clone(),
                RuntimeConfigState {
                    defaults: ChannelRuntimeDefaults {
                        default_provider: "test-provider".to_string(),
                        model: "hot-reloaded-model".to_string(),
                        temperature: 0.5,
                        api_key: None,
                        api_url: None,
                        reliability: crate::config::ReliabilityConfig::default(),
                    },
                    last_applied_stamp: None,
                },
            );
        }

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::clone(&provider),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("startup-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(provider_cache_seed)),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions {
                zeroclaw_dir: Some(temp.path().to_path_buf()),
                ..providers::ProviderRuntimeOptions::default()
            },
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-runtime-store-model".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "hello runtime defaults".to_string(),
                channel: "telegram".to_string(),
                timestamp: 4,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        {
            let mut cleanup_store = runtime_config_store()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cleanup_store.remove(&config_path);
        }

        assert_eq!(provider_impl.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider_impl
                .models
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_slice(),
            &["hot-reloaded-model".to_string()]
        );
    }

    #[tokio::test]
    async fn process_channel_message_respects_configured_max_tool_iterations_above_default() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(IterativeToolProvider {
                required_tool_iterations: 11,
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 12,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-iter-success".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-iter-success".to_string(),
                content: "Loop until done".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert!(!sent_messages.is_empty());
        let reply = sent_messages.last().unwrap();
        assert!(reply.starts_with("chat-iter-success:"));
        assert!(reply.contains("Completed after 11 tool iterations."));
        assert!(!reply.contains("⚠️ Error:"));
    }

    #[tokio::test]
    async fn process_channel_message_reports_configured_max_tool_iterations_limit() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(IterativeToolProvider {
                required_tool_iterations: 20,
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![Box::new(MockPriceTool)]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 3,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-iter-fail".to_string(),
                sender: "bob".to_string(),
                reply_target: "chat-iter-fail".to_string(),
                content: "Loop forever".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 2,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert!(!sent_messages.is_empty());
        let reply = sent_messages.last().unwrap();
        assert!(reply.starts_with("chat-iter-fail:"));
        assert!(reply.contains("⚠️ Error: Agent exceeded maximum tool iterations (3)"));
    }

    struct NoopMemory;

    #[async_trait::async_trait]
    impl Memory for NoopMemory {
        fn name(&self) -> &str {
            "noop"
        }

        async fn store(
            &self,
            _key: &str,
            _content: &str,
            _category: crate::memory::MemoryCategory,
            _session_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<crate::memory::MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn get(&self, _key: &str) -> anyhow::Result<Option<crate::memory::MemoryEntry>> {
            Ok(None)
        }

        async fn list(
            &self,
            _category: Option<&crate::memory::MemoryCategory>,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<crate::memory::MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn forget(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn count(&self) -> anyhow::Result<usize> {
            Ok(0)
        }

        async fn health_check(&self) -> bool {
            true
        }
    }

    struct RecallMemory;

    #[async_trait::async_trait]
    impl Memory for RecallMemory {
        fn name(&self) -> &str {
            "recall-memory"
        }

        async fn store(
            &self,
            _key: &str,
            _content: &str,
            _category: crate::memory::MemoryCategory,
            _session_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<crate::memory::MemoryEntry>> {
            Ok(vec![crate::memory::MemoryEntry {
                id: "entry-1".to_string(),
                key: "memory_key_1".to_string(),
                content: "Age is 45".to_string(),
                category: crate::memory::MemoryCategory::Conversation,
                timestamp: "2026-02-20T00:00:00Z".to_string(),
                session_id: None,
                score: Some(0.9),
            }])
        }

        async fn get(&self, _key: &str) -> anyhow::Result<Option<crate::memory::MemoryEntry>> {
            Ok(None)
        }

        async fn list(
            &self,
            _category: Option<&crate::memory::MemoryCategory>,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<crate::memory::MemoryEntry>> {
            Ok(Vec::new())
        }

        async fn forget(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn count(&self) -> anyhow::Result<usize> {
            Ok(1)
        }

        async fn health_check(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn message_dispatch_processes_messages_in_parallel() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(SlowProvider {
                delay: Duration::from_millis(250),
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(4);
        tx.send(traits::ChannelMessage {
            id: "1".to_string(),
            sender: "alice".to_string(),
            reply_target: "alice".to_string(),
            content: "hello".to_string(),
            channel: "test-channel".to_string(),
            timestamp: 1,
            thread_ts: None,
        })
        .await
        .unwrap();
        tx.send(traits::ChannelMessage {
            id: "2".to_string(),
            sender: "bob".to_string(),
            reply_target: "bob".to_string(),
            content: "world".to_string(),
            channel: "test-channel".to_string(),
            timestamp: 2,
            thread_ts: None,
        })
        .await
        .unwrap();
        drop(tx);

        let started = Instant::now();
        run_message_dispatch_loop(rx, runtime_ctx, 2).await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(430),
            "expected parallel dispatch (<430ms), got {:?}",
            elapsed
        );

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert_eq!(sent_messages.len(), 2);
    }

    #[tokio::test]
    async fn message_dispatch_interrupts_in_flight_telegram_request_and_preserves_context() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let provider_impl = Arc::new(DelayedHistoryCaptureProvider {
            delay: Duration::from_millis(250),
            calls: std::sync::Mutex::new(Vec::new()),
        });

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: provider_impl.clone(),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: true,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(8);
        let send_task = tokio::spawn(async move {
            tx.send(traits::ChannelMessage {
                id: "msg-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "forwarded content".to_string(),
                channel: "telegram".to_string(),
                timestamp: 1,
                thread_ts: None,
            })
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_millis(40)).await;
            tx.send(traits::ChannelMessage {
                id: "msg-2".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "summarize this".to_string(),
                channel: "telegram".to_string(),
                timestamp: 2,
                thread_ts: None,
            })
            .await
            .unwrap();
        });

        run_message_dispatch_loop(rx, runtime_ctx, 4).await;
        send_task.await.unwrap();

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert_eq!(sent_messages.len(), 1);
        assert!(sent_messages[0].starts_with("chat-1:"));
        assert!(sent_messages[0].contains("response-2"));
        drop(sent_messages);

        let calls = provider_impl
            .calls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(calls.len(), 2);
        let second_call = &calls[1];
        assert!(second_call
            .iter()
            .any(|(role, content)| { role == "user" && content.contains("forwarded content") }));
        assert!(second_call
            .iter()
            .any(|(role, content)| { role == "user" && content.contains("summarize this") }));
        assert!(
            !second_call.iter().any(|(role, _)| role == "assistant"),
            "cancelled turn should not persist an assistant response"
        );
    }

    #[tokio::test]
    async fn message_dispatch_interrupt_scope_is_same_sender_same_chat() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(SlowProvider {
                delay: Duration::from_millis(180),
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: true,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(8);
        let send_task = tokio::spawn(async move {
            tx.send(traits::ChannelMessage {
                id: "msg-a".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "first chat".to_string(),
                channel: "telegram".to_string(),
                timestamp: 1,
                thread_ts: None,
            })
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
            tx.send(traits::ChannelMessage {
                id: "msg-b".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-2".to_string(),
                content: "second chat".to_string(),
                channel: "telegram".to_string(),
                timestamp: 2,
                thread_ts: None,
            })
            .await
            .unwrap();
        });

        run_message_dispatch_loop(rx, runtime_ctx, 4).await;
        send_task.await.unwrap();

        let sent_messages = channel_impl.sent_messages.lock().await;
        assert_eq!(sent_messages.len(), 2);
        assert!(sent_messages.iter().any(|msg| msg.starts_with("chat-1:")));
        assert!(sent_messages.iter().any(|msg| msg.starts_with("chat-2:")));
    }

    #[tokio::test]
    async fn process_channel_message_cancels_scoped_typing_task() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(SlowProvider {
                delay: Duration::from_millis(20),
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "typing-msg".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-typing".to_string(),
                content: "hello".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let starts = channel_impl.start_typing_calls.load(Ordering::SeqCst);
        let stops = channel_impl.stop_typing_calls.load(Ordering::SeqCst);
        assert_eq!(starts, 1, "start_typing should be called once");
        assert_eq!(stops, 1, "stop_typing should be called once");
    }

    #[tokio::test]
    async fn process_channel_message_adds_and_swaps_reactions() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(SlowProvider {
                delay: Duration::from_millis(5),
            }),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 10,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "react-msg".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-react".to_string(),
                content: "hello".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let added = channel_impl.reactions_added.lock().await;
        assert!(
            added.len() >= 2,
            "expected at least 2 reactions added (\u{1F440} then \u{2705}), got {}",
            added.len()
        );
        assert_eq!(added[0].2, "\u{1F440}", "first reaction should be eyes");
        assert_eq!(
            added.last().unwrap().2,
            "\u{2705}",
            "last reaction should be checkmark"
        );

        let removed = channel_impl.reactions_removed.lock().await;
        assert_eq!(removed.len(), 1, "eyes reaction should be removed once");
        assert_eq!(removed[0].2, "\u{1F440}");
    }

    #[test]
    fn prompt_contains_all_sections() {
        let ws = make_workspace();
        let tools = vec![("shell", "Run commands"), ("file_read", "Read files")];
        let prompt = build_system_prompt(ws.path(), "test-model", &tools, &[], None, None);

        // Section headers
        assert!(prompt.contains("## Tools"), "missing Tools section");
        assert!(prompt.contains("## Safety"), "missing Safety section");
        assert!(prompt.contains("## Workspace"), "missing Workspace section");
        assert!(
            prompt.contains("## Project Context"),
            "missing Project Context"
        );
        assert!(
            prompt.contains("## Current Date & Time"),
            "missing Date/Time"
        );
        assert!(prompt.contains("## Runtime"), "missing Runtime section");
    }

    #[test]
    fn prompt_injects_tools() {
        let ws = make_workspace();
        let tools = vec![
            ("shell", "Run commands"),
            ("memory_recall", "Search memory"),
        ];
        let prompt = build_system_prompt(ws.path(), "gpt-4o", &tools, &[], None, None);

        assert!(prompt.contains("**shell**"));
        assert!(prompt.contains("Run commands"));
        assert!(prompt.contains("**memory_recall**"));
    }

    #[test]
    fn prompt_includes_single_tool_protocol_block_after_append() {
        let ws = make_workspace();
        let tools = vec![("shell", "Run commands")];
        let mut prompt = build_system_prompt(ws.path(), "gpt-4o", &tools, &[], None, None);

        assert!(
            !prompt.contains("## Tool Use Protocol"),
            "build_system_prompt should not emit protocol block directly"
        );

        prompt.push_str(&build_tool_instructions(&[]));

        assert_eq!(
            prompt.matches("## Tool Use Protocol").count(),
            1,
            "protocol block should appear exactly once in the final prompt"
        );
    }

    #[test]
    fn prompt_injects_safety() {
        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        assert!(prompt.contains("Do not exfiltrate private data"));
        assert!(prompt.contains("Do not run destructive commands"));
        assert!(prompt.contains("Prefer `trash` over `rm`"));
    }

    #[test]
    fn prompt_injects_workspace_files() {
        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        assert!(prompt.contains("### SOUL.md"), "missing SOUL.md header");
        assert!(prompt.contains("Be helpful"), "missing SOUL content");
        assert!(prompt.contains("### IDENTITY.md"), "missing IDENTITY.md");
        assert!(
            prompt.contains("Name: ZeroClaw"),
            "missing IDENTITY content"
        );
        assert!(prompt.contains("### USER.md"), "missing USER.md");
        assert!(prompt.contains("### AGENTS.md"), "missing AGENTS.md");
        assert!(prompt.contains("### TOOLS.md"), "missing TOOLS.md");
        // HEARTBEAT.md is intentionally excluded from channel prompts — it's only
        // relevant to the heartbeat worker and causes LLMs to emit spurious
        // "HEARTBEAT_OK" acknowledgments in channel conversations.
        assert!(
            !prompt.contains("### HEARTBEAT.md"),
            "HEARTBEAT.md should not be in channel prompt"
        );
        assert!(prompt.contains("### MEMORY.md"), "missing MEMORY.md");
        assert!(prompt.contains("User likes Rust"), "missing MEMORY content");
    }

    #[test]
    fn prompt_missing_file_markers() {
        let tmp = TempDir::new().unwrap();
        // Empty workspace — no files at all
        let prompt = build_system_prompt(tmp.path(), "model", &[], &[], None, None);

        assert!(prompt.contains("[File not found: SOUL.md]"));
        assert!(prompt.contains("[File not found: AGENTS.md]"));
        assert!(prompt.contains("[File not found: IDENTITY.md]"));
    }

    #[test]
    fn prompt_bootstrap_only_if_exists() {
        let ws = make_workspace();
        // No BOOTSTRAP.md — should not appear
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);
        assert!(
            !prompt.contains("### BOOTSTRAP.md"),
            "BOOTSTRAP.md should not appear when missing"
        );

        // Create BOOTSTRAP.md — should appear
        std::fs::write(ws.path().join("BOOTSTRAP.md"), "# Bootstrap\nFirst run.").unwrap();
        let prompt2 = build_system_prompt(ws.path(), "model", &[], &[], None, None);
        assert!(
            prompt2.contains("### BOOTSTRAP.md"),
            "BOOTSTRAP.md should appear when present"
        );
        assert!(prompt2.contains("First run"));
    }

    #[test]
    fn prompt_no_daily_memory_injection() {
        let ws = make_workspace();
        let memory_dir = ws.path().join("memory");
        std::fs::create_dir_all(&memory_dir).unwrap();
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        std::fs::write(
            memory_dir.join(format!("{today}.md")),
            "# Daily\nSome note.",
        )
        .unwrap();

        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        // Daily notes should NOT be in the system prompt (on-demand via tools)
        assert!(
            !prompt.contains("Daily Notes"),
            "daily notes should not be auto-injected"
        );
        assert!(
            !prompt.contains("Some note"),
            "daily content should not be in prompt"
        );
    }

    #[test]
    fn prompt_runtime_metadata() {
        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "claude-sonnet-4", &[], &[], None, None);

        assert!(prompt.contains("Model: claude-sonnet-4"));
        assert!(prompt.contains(&format!("OS: {}", std::env::consts::OS)));
        assert!(prompt.contains("Host:"));
    }

    #[test]
    fn prompt_skills_include_instructions_and_tools() {
        let ws = make_workspace();
        let skills = vec![crate::skills::Skill {
            name: "code-review".into(),
            description: "Review code for bugs".into(),
            version: "1.0.0".into(),
            author: None,
            tags: vec![],
            tools: vec![crate::skills::SkillTool {
                name: "lint".into(),
                description: "Run static checks".into(),
                kind: "shell".into(),
                command: "cargo clippy".into(),
                args: HashMap::new(),
            }],
            prompts: vec!["Always run cargo test before final response.".into()],
            location: None,
        }];

        let prompt = build_system_prompt(ws.path(), "model", &[], &skills, None, None);

        assert!(prompt.contains("<available_skills>"), "missing skills XML");
        assert!(prompt.contains("<name>code-review</name>"));
        assert!(prompt.contains("<description>Review code for bugs</description>"));
        assert!(prompt.contains("SKILL.md</location>"));
        assert!(prompt.contains("<instructions>"));
        assert!(prompt
            .contains("<instruction>Always run cargo test before final response.</instruction>"));
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("<name>lint</name>"));
        assert!(prompt.contains("<kind>shell</kind>"));
        assert!(!prompt.contains("loaded on demand"));
    }

    #[test]
    fn prompt_skills_compact_mode_omits_instructions_and_tools() {
        let ws = make_workspace();
        let skills = vec![crate::skills::Skill {
            name: "code-review".into(),
            description: "Review code for bugs".into(),
            version: "1.0.0".into(),
            author: None,
            tags: vec![],
            tools: vec![crate::skills::SkillTool {
                name: "lint".into(),
                description: "Run static checks".into(),
                kind: "shell".into(),
                command: "cargo clippy".into(),
                args: HashMap::new(),
            }],
            prompts: vec!["Always run cargo test before final response.".into()],
            location: None,
        }];

        let prompt = build_system_prompt_with_mode(
            ws.path(),
            "model",
            &[],
            &skills,
            None,
            None,
            false,
            crate::config::SkillsPromptInjectionMode::Compact,
        );

        assert!(prompt.contains("<available_skills>"), "missing skills XML");
        assert!(prompt.contains("<name>code-review</name>"));
        assert!(prompt.contains("<location>skills/code-review/SKILL.md</location>"));
        assert!(prompt.contains("loaded on demand"));
        assert!(!prompt.contains("<instructions>"));
        assert!(!prompt
            .contains("<instruction>Always run cargo test before final response.</instruction>"));
        assert!(!prompt.contains("<tools>"));
    }

    #[test]
    fn prompt_skills_escape_reserved_xml_chars() {
        let ws = make_workspace();
        let skills = vec![crate::skills::Skill {
            name: "code<review>&".into(),
            description: "Review \"unsafe\" and 'risky' bits".into(),
            version: "1.0.0".into(),
            author: None,
            tags: vec![],
            tools: vec![crate::skills::SkillTool {
                name: "run\"linter\"".into(),
                description: "Run <lint> & report".into(),
                kind: "shell&exec".into(),
                command: "cargo clippy".into(),
                args: HashMap::new(),
            }],
            prompts: vec!["Use <tool_call> and & keep output \"safe\"".into()],
            location: None,
        }];

        let prompt = build_system_prompt(ws.path(), "model", &[], &skills, None, None);

        assert!(prompt.contains("<name>code&lt;review&gt;&amp;</name>"));
        assert!(prompt.contains(
            "<description>Review &quot;unsafe&quot; and &apos;risky&apos; bits</description>"
        ));
        assert!(prompt.contains("<name>run&quot;linter&quot;</name>"));
        assert!(prompt.contains("<description>Run &lt;lint&gt; &amp; report</description>"));
        assert!(prompt.contains("<kind>shell&amp;exec</kind>"));
        assert!(prompt.contains(
            "<instruction>Use &lt;tool_call&gt; and &amp; keep output &quot;safe&quot;</instruction>"
        ));
    }

    #[test]
    fn prompt_truncation() {
        let ws = make_workspace();
        // Write a file larger than BOOTSTRAP_MAX_CHARS
        let big_content = "x".repeat(BOOTSTRAP_MAX_CHARS + 1000);
        std::fs::write(ws.path().join("AGENTS.md"), &big_content).unwrap();

        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        assert!(
            prompt.contains("truncated at"),
            "large files should be truncated"
        );
        assert!(
            !prompt.contains(&big_content),
            "full content should not appear"
        );
    }

    #[test]
    fn prompt_empty_files_skipped() {
        let ws = make_workspace();
        std::fs::write(ws.path().join("TOOLS.md"), "").unwrap();

        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        // Empty file should not produce a header
        assert!(
            !prompt.contains("### TOOLS.md"),
            "empty files should be skipped"
        );
    }

    #[test]
    fn channel_log_truncation_is_utf8_safe_for_multibyte_text() {
        let msg = "Hello from ZeroClaw 🌍. Current status is healthy, and café-style UTF-8 text stays safe in logs.";

        // Reproduces the production crash path where channel logs truncate at 80 chars.
        let result = std::panic::catch_unwind(|| crate::util::truncate_with_ellipsis(msg, 80));
        assert!(
            result.is_ok(),
            "truncate_with_ellipsis should never panic on UTF-8"
        );

        let truncated = result.unwrap();
        assert!(!truncated.is_empty());
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn prompt_contains_channel_capabilities() {
        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        assert!(
            prompt.contains("## Channel Capabilities"),
            "missing Channel Capabilities section"
        );
        assert!(
            prompt.contains("running as a messaging bot"),
            "missing channel context"
        );
        assert!(
            prompt.contains("NEVER repeat, describe, or echo credentials"),
            "missing security instruction"
        );
    }

    #[test]
    fn prompt_workspace_path() {
        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        assert!(prompt.contains(&format!("Working directory: `{}`", ws.path().display())));
    }

    #[test]
    fn conversation_memory_key_uses_message_id() {
        let msg = traits::ChannelMessage {
            id: "msg_abc123".into(),
            sender: "U123".into(),
            reply_target: "C456".into(),
            content: "hello".into(),
            channel: "slack".into(),
            timestamp: 1,
            thread_ts: None,
        };

        assert_eq!(conversation_memory_key(&msg), "slack_U123_msg_abc123");
    }

    #[test]
    fn conversation_memory_key_is_unique_per_message() {
        let msg1 = traits::ChannelMessage {
            id: "msg_1".into(),
            sender: "U123".into(),
            reply_target: "C456".into(),
            content: "first".into(),
            channel: "slack".into(),
            timestamp: 1,
            thread_ts: None,
        };
        let msg2 = traits::ChannelMessage {
            id: "msg_2".into(),
            sender: "U123".into(),
            reply_target: "C456".into(),
            content: "second".into(),
            channel: "slack".into(),
            timestamp: 2,
            thread_ts: None,
        };

        assert_ne!(
            conversation_memory_key(&msg1),
            conversation_memory_key(&msg2)
        );
    }

    #[tokio::test]
    async fn autosave_keys_preserve_multiple_conversation_facts() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();

        let msg1 = traits::ChannelMessage {
            id: "msg_1".into(),
            sender: "U123".into(),
            reply_target: "C456".into(),
            content: "I'm Paul".into(),
            channel: "slack".into(),
            timestamp: 1,
            thread_ts: None,
        };
        let msg2 = traits::ChannelMessage {
            id: "msg_2".into(),
            sender: "U123".into(),
            reply_target: "C456".into(),
            content: "I'm 45".into(),
            channel: "slack".into(),
            timestamp: 2,
            thread_ts: None,
        };

        mem.store(
            &conversation_memory_key(&msg1),
            &msg1.content,
            MemoryCategory::Conversation,
            None,
        )
        .await
        .unwrap();
        mem.store(
            &conversation_memory_key(&msg2),
            &msg2.content,
            MemoryCategory::Conversation,
            None,
        )
        .await
        .unwrap();

        assert_eq!(mem.count().await.unwrap(), 2);

        let recalled = mem.recall("45", 5, None).await.unwrap();
        assert!(recalled.iter().any(|entry| entry.content.contains("45")));
    }

    #[tokio::test]
    async fn build_memory_context_includes_recalled_entries() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();
        mem.store("age_fact", "Age is 45", MemoryCategory::Conversation, None)
            .await
            .unwrap();

        let context = build_memory_context(&mem, "age", 0.0).await;
        assert!(context.contains("[Memory context]"));
        assert!(context.contains("Age is 45"));
    }

    /// Auto-saved photo messages must not surface through memory context,
    /// otherwise the image marker gets duplicated in the provider request (#2403).
    #[tokio::test]
    async fn build_memory_context_excludes_image_marker_entries() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();

        // Simulate auto-save of a photo message containing an [IMAGE:] marker.
        mem.store(
            "telegram_user_msg_photo",
            "[IMAGE:/tmp/workspace/photo_1_2.jpg]\n\nDescribe this screenshot",
            MemoryCategory::Conversation,
            None,
        )
        .await
        .unwrap();
        // Also store a plain text entry that shares a word with the query
        // so the FTS recall returns both entries.
        mem.store(
            "screenshot_preference",
            "User prefers screenshot descriptions to be concise",
            MemoryCategory::Conversation,
            None,
        )
        .await
        .unwrap();

        let context = build_memory_context(&mem, "screenshot", 0.0).await;

        // The image-marker entry must be excluded to prevent duplication.
        assert!(
            !context.contains("[IMAGE:"),
            "memory context must not contain image markers, got: {context}"
        );
        // Plain text entries should still be included.
        assert!(
            context.contains("screenshot descriptions"),
            "plain text entry should remain in context, got: {context}"
        );
    }

    #[tokio::test]
    async fn process_channel_message_restores_per_sender_history_on_follow_ups() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let provider_impl = Arc::new(HistoryCaptureProvider::default());

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: provider_impl.clone(),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx.clone(),
            traits::ChannelMessage {
                id: "msg-a".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "hello".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-b".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-1".to_string(),
                content: "follow up".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 2,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let calls = provider_impl
            .calls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[0][0].0, "system");
        assert_eq!(calls[0][1].0, "user");
        assert_eq!(calls[1].len(), 4);
        assert_eq!(calls[1][0].0, "system");
        assert_eq!(calls[1][1].0, "user");
        assert_eq!(calls[1][2].0, "assistant");
        assert_eq!(calls[1][3].0, "user");
        assert!(calls[1][1].1.contains("hello"));
        assert!(calls[1][2].1.contains("response-1"));
        assert!(calls[1][3].1.contains("follow up"));
    }

    #[tokio::test]
    async fn process_channel_message_enriches_current_turn_without_persisting_context() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let provider_impl = Arc::new(HistoryCaptureProvider::default());
        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: provider_impl.clone(),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(RecallMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx.clone(),
            traits::ChannelMessage {
                id: "msg-ctx-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-ctx".to_string(),
                content: "hello".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let calls = provider_impl
            .calls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[0][1].0, "user");
        assert!(calls[0][1].1.contains("[Memory context]"));
        assert!(calls[0][1].1.contains("Age is 45"));
        assert!(calls[0][1].1.contains("hello"));

        let histories = runtime_ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let turns = histories
            .get("test-channel_alice")
            .expect("history should be stored for sender");
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content, "hello");
        assert!(!turns[0].content.contains("[Memory context]"));
    }

    #[tokio::test]
    async fn process_channel_message_telegram_keeps_system_instruction_at_top_only() {
        let channel_impl = Arc::new(TelegramRecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let provider_impl = Arc::new(HistoryCaptureProvider::default());
        let mut histories = HashMap::new();
        histories.insert(
            "telegram_alice".to_string(),
            vec![
                ChatMessage::assistant("stale assistant"),
                ChatMessage::user("earlier user question"),
                ChatMessage::assistant("earlier assistant reply"),
            ],
        );

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: provider_impl.clone(),
            default_provider: Arc::new("test-provider".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("test-system-prompt".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(histories)),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            runtime_ctx.clone(),
            traits::ChannelMessage {
                id: "tg-msg-1".to_string(),
                sender: "alice".to_string(),
                reply_target: "chat-telegram".to_string(),
                content: "hello".to_string(),
                channel: "telegram".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let calls = provider_impl
            .calls
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 4);

        let roles = calls[0]
            .iter()
            .map(|(role, _)| role.as_str())
            .collect::<Vec<_>>();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
        assert!(
            calls[0][0].1.contains("When responding on Telegram:"),
            "telegram channel instructions should be embedded into the system prompt"
        );
        assert!(
            calls[0][0].1.contains("For media attachments use markers:"),
            "telegram media marker guidance should live in the system prompt"
        );
        assert!(!calls[0].iter().skip(1).any(|(role, _)| role == "system"));
    }

    #[test]
    fn extract_tool_context_summary_collects_alias_and_native_tool_calls() {
        let history = vec![
            ChatMessage::system("sys"),
            ChatMessage::assistant(
                r#"<toolcall>
{"name":"shell","arguments":{"command":"date"}}
</toolcall>"#,
            ),
            ChatMessage::assistant(
                r#"{"content":null,"tool_calls":[{"id":"1","name":"web_search","arguments":"{}"}]}"#,
            ),
        ];

        let summary = extract_tool_context_summary(&history, 1);
        assert_eq!(summary, "[Used tools: shell, web_search]");
    }

    #[test]
    fn extract_tool_context_summary_collects_prompt_mode_tool_result_names() {
        let history = vec![
            ChatMessage::system("sys"),
            ChatMessage::assistant("Using markdown tool call fence"),
            ChatMessage::user(
                r#"[Tool results]
<tool_result name="http_request">
{"status":200}
</tool_result>
<tool_result name="shell">
Mon Feb 20
</tool_result>"#,
            ),
        ];

        let summary = extract_tool_context_summary(&history, 1);
        assert_eq!(summary, "[Used tools: http_request, shell]");
    }

    #[test]
    fn extract_tool_context_summary_respects_start_index() {
        let history = vec![
            ChatMessage::assistant(
                r#"<tool_call>
{"name":"stale_tool","arguments":{}}
</tool_call>"#,
            ),
            ChatMessage::assistant(
                r#"<tool_call>
{"name":"fresh_tool","arguments":{}}
</tool_call>"#,
            ),
        ];

        let summary = extract_tool_context_summary(&history, 1);
        assert_eq!(summary, "[Used tools: fresh_tool]");
    }

    #[test]
    fn strip_isolated_tool_json_artifacts_removes_tool_calls_and_results() {
        let mut known_tools = HashSet::new();
        known_tools.insert("schedule".to_string());

        let input = r#"{"name":"schedule","parameters":{"action":"create","message":"test"}}
{"name":"schedule","parameters":{"action":"cancel","task_id":"test"}}
Let me create the reminder properly:
{"name":"schedule","parameters":{"action":"create","message":"Go to sleep"}}
{"result":{"task_id":"abc","status":"scheduled"}}
Done reminder set for 1:38 AM."#;

        let result = strip_isolated_tool_json_artifacts(input, &known_tools);
        let normalized = result
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            normalized,
            "Let me create the reminder properly:\nDone reminder set for 1:38 AM."
        );
    }

    #[test]
    fn strip_isolated_tool_json_artifacts_preserves_non_tool_json() {
        let mut known_tools = HashSet::new();
        known_tools.insert("shell".to_string());

        let input = r#"{"name":"profile","parameters":{"timezone":"UTC"}}
This is an example JSON object for profile settings."#;

        let result = strip_isolated_tool_json_artifacts(input, &known_tools);
        assert_eq!(result, input);
    }

    // ── AIEOS Identity Tests (Issue #168) ─────────────────────────

    #[test]
    fn aieos_identity_from_file() {
        use crate::config::IdentityConfig;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let identity_path = tmp.path().join("aieos_identity.json");

        // Write AIEOS identity file
        let aieos_json = r#"{
            "identity": {
                "names": {"first": "Nova", "nickname": "Nov"},
                "bio": "A helpful AI assistant.",
                "origin": "Silicon Valley"
            },
            "psychology": {
                "mbti": "INTJ",
                "moral_compass": ["Be helpful", "Do no harm"]
            },
            "linguistics": {
                "style": "concise",
                "formality": "casual"
            }
        }"#;
        std::fs::write(&identity_path, aieos_json).unwrap();

        // Create identity config pointing to the file
        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: Some("aieos_identity.json".into()),
            aieos_inline: None,
        };

        let prompt = build_system_prompt(tmp.path(), "model", &[], &[], Some(&config), None);

        // Should contain AIEOS sections
        assert!(prompt.contains("## Identity"));
        assert!(prompt.contains("**Name:** Nova"));
        assert!(prompt.contains("**Nickname:** Nov"));
        assert!(prompt.contains("**Bio:** A helpful AI assistant."));
        assert!(prompt.contains("**Origin:** Silicon Valley"));

        assert!(prompt.contains("## Personality"));
        assert!(prompt.contains("**MBTI:** INTJ"));
        assert!(prompt.contains("**Moral Compass:**"));
        assert!(prompt.contains("- Be helpful"));

        assert!(prompt.contains("## Communication Style"));
        assert!(prompt.contains("**Style:** concise"));
        assert!(prompt.contains("**Formality Level:** casual"));

        // Should NOT contain OpenClaw bootstrap file headers
        assert!(!prompt.contains("### SOUL.md"));
        assert!(!prompt.contains("### IDENTITY.md"));
        assert!(!prompt.contains("[File not found"));
    }

    #[test]
    fn aieos_identity_from_inline() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: None,
            aieos_inline: Some(r#"{"identity":{"names":{"first":"Claw"}}}"#.into()),
        };

        let prompt = build_system_prompt(
            std::env::temp_dir().as_path(),
            "model",
            &[],
            &[],
            Some(&config),
            None,
        );

        assert!(prompt.contains("**Name:** Claw"));
        assert!(prompt.contains("## Identity"));
    }

    #[test]
    fn aieos_fallback_to_openclaw_on_parse_error() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: Some("nonexistent.json".into()),
            aieos_inline: None,
        };

        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], Some(&config), None);

        // Should fall back to OpenClaw format when AIEOS file is not found
        // (Error is logged to stderr with filename, not included in prompt)
        assert!(prompt.contains("### SOUL.md"));
    }

    #[test]
    fn aieos_empty_uses_openclaw() {
        use crate::config::IdentityConfig;

        // Format is "aieos" but neither path nor inline is set
        let config = IdentityConfig {
            format: "aieos".into(),
            aieos_path: None,
            aieos_inline: None,
        };

        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], Some(&config), None);

        // Should use OpenClaw format (not configured for AIEOS)
        assert!(prompt.contains("### SOUL.md"));
        assert!(prompt.contains("Be helpful"));
    }

    #[test]
    fn openclaw_format_uses_bootstrap_files() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            format: "openclaw".into(),
            aieos_path: Some("identity.json".into()),
            aieos_inline: None,
        };

        let ws = make_workspace();
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], Some(&config), None);

        // Should use OpenClaw format even if aieos_path is set
        assert!(prompt.contains("### SOUL.md"));
        assert!(prompt.contains("Be helpful"));
        assert!(!prompt.contains("## Identity"));
    }

    #[test]
    fn none_identity_config_uses_openclaw() {
        let ws = make_workspace();
        // Pass None for identity config
        let prompt = build_system_prompt(ws.path(), "model", &[], &[], None, None);

        // Should use OpenClaw format
        assert!(prompt.contains("### SOUL.md"));
        assert!(prompt.contains("Be helpful"));
    }

    #[test]
    fn classify_health_ok_true() {
        let state = classify_health_result(&Ok(true));
        assert_eq!(state, ChannelHealthState::Healthy);
    }

    #[test]
    fn classify_health_ok_false() {
        let state = classify_health_result(&Ok(false));
        assert_eq!(state, ChannelHealthState::Unhealthy);
    }

    #[tokio::test]
    async fn classify_health_timeout() {
        let result = tokio::time::timeout(Duration::from_millis(1), async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            true
        })
        .await;
        let state = classify_health_result(&result);
        assert_eq!(state, ChannelHealthState::Timeout);
    }

    #[test]
    fn collect_configured_channels_includes_mattermost_when_configured() {
        let mut config = Config::default();
        config.channels_config.mattermost = Some(crate::config::schema::MattermostConfig {
            url: "https://mattermost.example.com".to_string(),
            bot_token: "test-token".to_string(),
            channel_id: Some("channel-1".to_string()),
            allowed_users: vec![],
            thread_replies: Some(true),
            mention_only: Some(false),
        });

        let channels = collect_configured_channels(&config, "test");

        assert!(channels
            .iter()
            .any(|entry| entry.display_name == "Mattermost"));
        assert!(channels
            .iter()
            .any(|entry| entry.channel.name() == "mattermost"));
    }

    struct AlwaysFailChannel {
        name: &'static str,
        calls: Arc<AtomicUsize>,
    }

    struct BlockUntilClosedChannel {
        name: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Channel for AlwaysFailChannel {
        fn name(&self) -> &str {
            self.name
        }

        async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
        ) -> anyhow::Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("listen boom")
        }
    }

    #[async_trait::async_trait]
    impl Channel for BlockUntilClosedChannel {
        fn name(&self) -> &str {
            &self.name
        }

        async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn listen(
            &self,
            tx: tokio::sync::mpsc::Sender<traits::ChannelMessage>,
        ) -> anyhow::Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tx.closed().await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn supervised_listener_marks_error_and_restarts_on_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let channel: Arc<dyn Channel> = Arc::new(AlwaysFailChannel {
            name: "test-supervised-fail",
            calls: Arc::clone(&calls),
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(1);
        let handle = spawn_supervised_listener(channel, tx, 1, 1);

        tokio::time::sleep(Duration::from_millis(80)).await;
        drop(rx);
        handle.abort();
        let _ = handle.await;

        let snapshot = crate::health::snapshot_json();
        let component = &snapshot["components"]["channel:test-supervised-fail"];
        assert_eq!(component["status"], "error");
        assert!(component["restart_count"].as_u64().unwrap_or(0) >= 1);
        assert!(component["last_error"]
            .as_str()
            .unwrap_or("")
            .contains("listen boom"));
        assert!(calls.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn supervised_listener_refreshes_health_while_running() {
        let calls = Arc::new(AtomicUsize::new(0));
        let channel_name = format!("test-supervised-heartbeat-{}", uuid::Uuid::new_v4());
        let component_name = format!("channel:{channel_name}");
        let channel: Arc<dyn Channel> = Arc::new(BlockUntilClosedChannel {
            name: channel_name,
            calls: Arc::clone(&calls),
        });

        let (tx, rx) = tokio::sync::mpsc::channel::<traits::ChannelMessage>(1);
        let handle = spawn_supervised_listener_with_health_interval(
            channel,
            tx,
            1,
            1,
            Duration::from_millis(20),
        );

        tokio::time::sleep(Duration::from_millis(35)).await;
        let first_last_ok = crate::health::snapshot_json()["components"][&component_name]
            ["last_ok"]
            .as_str()
            .unwrap_or("")
            .to_string();
        assert!(!first_last_ok.is_empty());

        tokio::time::sleep(Duration::from_millis(70)).await;
        let second_last_ok = crate::health::snapshot_json()["components"][&component_name]
            ["last_ok"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let first = chrono::DateTime::parse_from_rfc3339(&first_last_ok)
            .expect("last_ok should be valid RFC3339");
        let second = chrono::DateTime::parse_from_rfc3339(&second_last_ok)
            .expect("last_ok should be valid RFC3339");
        assert!(second > first, "expected periodic health heartbeat refresh");

        drop(rx);
        let join = tokio::time::timeout(Duration::from_secs(1), handle).await;
        assert!(join.is_ok(), "listener should stop after channel shutdown");
        assert!(calls.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn maybe_restart_daemon_systemd_args_regression() {
        assert_eq!(
            SYSTEMD_STATUS_ARGS,
            ["--user", "is-active", "zeroclaw.service"]
        );
        assert_eq!(
            SYSTEMD_RESTART_ARGS,
            ["--user", "restart", "zeroclaw.service"]
        );
    }

    #[test]
    fn maybe_restart_daemon_openrc_args_regression() {
        assert_eq!(OPENRC_STATUS_ARGS, ["zeroclaw", "status"]);
        assert_eq!(OPENRC_RESTART_ARGS, ["zeroclaw", "restart"]);
    }

    #[test]
    fn normalize_merges_consecutive_user_turns() {
        let turns = vec![ChatMessage::user("hello"), ChatMessage::user("world")];
        let result = normalize_cached_channel_turns(turns);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content, "hello\n\nworld");
    }

    #[test]
    fn normalize_preserves_strict_alternation() {
        let turns = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
            ChatMessage::user("bye"),
        ];
        let result = normalize_cached_channel_turns(turns);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "hello");
        assert_eq!(result[1].content, "hi");
        assert_eq!(result[2].content, "bye");
    }

    #[test]
    fn normalize_merges_multiple_consecutive_user_turns() {
        let turns = vec![
            ChatMessage::user("a"),
            ChatMessage::user("b"),
            ChatMessage::user("c"),
        ];
        let result = normalize_cached_channel_turns(turns);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content, "a\n\nb\n\nc");
    }

    #[test]
    fn normalize_empty_input() {
        let result = normalize_cached_channel_turns(vec![]);
        assert!(result.is_empty());
    }

    // ── E2E: photo [IMAGE:] marker rejected by non-vision provider ───

    /// End-to-end test: a photo attachment message (containing `[IMAGE:]`
    /// marker) sent through `process_channel_message` with a non-vision
    /// provider must produce a `"⚠️ Error: …does not support vision"` reply
    /// on the recording channel — no real Telegram or LLM API required.
    #[tokio::test]
    async fn e2e_photo_attachment_rejected_by_non_vision_provider() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        // DummyProvider has default capabilities (vision: false).
        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(DummyProvider),
            default_provider: Arc::new("dummy".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("You are a helpful assistant.".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        // Simulate a photo attachment message with [IMAGE:] marker.
        process_channel_message(
            runtime_ctx,
            traits::ChannelMessage {
                id: "msg-photo-1".to_string(),
                sender: "zeroclaw_user".to_string(),
                reply_target: "chat-photo".to_string(),
                content: "[IMAGE:/tmp/workspace/photo_99_1.jpg]\n\nWhat is this?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent = channel_impl.sent_messages.lock().await;
        assert_eq!(sent.len(), 1, "expected exactly one reply message");
        assert!(
            sent[0].contains("does not support vision"),
            "reply must mention vision capability error, got: {}",
            sent[0]
        );
        assert!(
            sent[0].contains("⚠️ Error"),
            "reply must start with error prefix, got: {}",
            sent[0]
        );
    }

    #[tokio::test]
    async fn e2e_failed_vision_turn_does_not_poison_follow_up_text_turn() {
        let channel_impl = Arc::new(RecordingChannel::default());
        let channel: Arc<dyn Channel> = channel_impl.clone();

        let mut channels_by_name = HashMap::new();
        channels_by_name.insert(channel.name().to_string(), channel);

        let runtime_ctx = Arc::new(ChannelRuntimeContext {
            channels_by_name: Arc::new(channels_by_name),
            provider: Arc::new(DummyProvider),
            default_provider: Arc::new("dummy".to_string()),
            memory: Arc::new(NoopMemory),
            tools_registry: Arc::new(vec![]),
            observer: Arc::new(NoopObserver),
            system_prompt: Arc::new("You are a helpful assistant.".to_string()),
            model: Arc::new("test-model".to_string()),
            temperature: 0.0,
            auto_save_memory: false,
            max_tool_iterations: 5,
            min_relevance_score: 0.0,
            conversation_histories: Arc::new(Mutex::new(HashMap::new())),
            provider_cache: Arc::new(Mutex::new(HashMap::new())),
            route_overrides: Arc::new(Mutex::new(HashMap::new())),
            api_key: None,
            api_url: None,
            reliability: Arc::new(crate::config::ReliabilityConfig::default()),
            provider_runtime_options: providers::ProviderRuntimeOptions::default(),
            workspace_dir: Arc::new(std::env::temp_dir()),
            message_timeout_secs: CHANNEL_MESSAGE_TIMEOUT_SECS,
            interrupt_on_new_message: false,
            multimodal: crate::config::MultimodalConfig::default(),
            hooks: None,
            non_cli_excluded_tools: Arc::new(Vec::new()),
            tool_call_dedup_exempt: Arc::new(Vec::new()),
            model_routes: Arc::new(Vec::new()),
        });

        process_channel_message(
            Arc::clone(&runtime_ctx),
            traits::ChannelMessage {
                id: "msg-photo-1".to_string(),
                sender: "zeroclaw_user".to_string(),
                reply_target: "chat-photo".to_string(),
                content: "[IMAGE:/tmp/workspace/photo_99_1.jpg]\n\nWhat is this?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 1,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        process_channel_message(
            Arc::clone(&runtime_ctx),
            traits::ChannelMessage {
                id: "msg-text-2".to_string(),
                sender: "zeroclaw_user".to_string(),
                reply_target: "chat-photo".to_string(),
                content: "What is WAL?".to_string(),
                channel: "test-channel".to_string(),
                timestamp: 2,
                thread_ts: None,
            },
            CancellationToken::new(),
        )
        .await;

        let sent = channel_impl.sent_messages.lock().await;
        assert_eq!(sent.len(), 2, "expected one error and one successful reply");
        assert!(
            sent[0].contains("does not support vision"),
            "first reply must mention vision capability error, got: {}",
            sent[0]
        );
        assert!(
            sent[1].ends_with(":ok"),
            "second reply should succeed for text-only turn, got: {}",
            sent[1]
        );
        drop(sent);

        let histories = runtime_ctx
            .conversation_histories
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let turns = histories
            .get("test-channel_zeroclaw_user")
            .expect("history should exist for sender");
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content, "What is WAL?");
        assert_eq!(turns[1].role, "assistant");
        assert_eq!(turns[1].content, "ok");
        assert!(
            turns.iter().all(|turn| !turn.content.contains("[IMAGE:")),
            "failed vision turn must not persist image marker content"
        );
    }

    #[test]
    fn build_channel_by_id_unknown_channel_returns_error() {
        let config = Config::default();
        match build_channel_by_id(&config, "nonexistent") {
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("Unknown channel"),
                    "expected 'Unknown channel' in error, got: {err_msg}"
                );
            }
            Ok(_) => panic!("should fail for unknown channel"),
        }
    }

    #[test]
    fn build_channel_by_id_unconfigured_telegram_returns_error() {
        let config = Config::default();
        match build_channel_by_id(&config, "telegram") {
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("not configured"),
                    "expected 'not configured' in error, got: {err_msg}"
                );
            }
            Ok(_) => panic!("should fail when telegram is not configured"),
        }
    }

    #[test]
    fn build_channel_by_id_configured_telegram_succeeds() {
        let mut config = Config::default();
        config.channels_config.telegram = Some(crate::config::schema::TelegramConfig {
            bot_token: "test-token".to_string(),
            allowed_users: vec![],
            stream_mode: crate::config::StreamMode::Off,
            draft_update_interval_ms: 1000,
            interrupt_on_new_message: false,
            mention_only: false,
        });
        match build_channel_by_id(&config, "telegram") {
            Ok(channel) => assert_eq!(channel.name(), "telegram"),
            Err(e) => panic!("should succeed when telegram is configured: {e}"),
        }
    }
}
