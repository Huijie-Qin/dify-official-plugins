package com.dify.agent_strategies.cot_agent.strategies;

import lombok.Data;

@Data
public class AgentSession {
    private ModelService model;
    private ToolService tool;

    @Data
    public static class ModelService {
        private LlmService llm;
    }

    @Data
    public static class LlmService {
        public Object invoke(
            LLMModelConfig modelConfig,
            Iterable<PromptMessage> promptMessages,
            boolean stream,
            Iterable<String> stop
        ) {
            return null;
        }

        public Object invoke(
            LLMModelConfig modelConfig,
            Iterable<PromptMessage> promptMessages,
            Iterable<String> stop,
            boolean stream,
            Iterable<ToolEntity> tools
        ) {
            return null;
        }
    }

    @Data
    public static class ToolService {
        public Iterable<ToolInvokeMessage> invoke(
            ToolProviderType providerType,
            String provider,
            String toolName,
            java.util.Map<String, Object> parameters
        ) {
            return java.util.List.of();
        }
    }
}
