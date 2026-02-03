package com.dify.agent_strategies.cot_agent.strategies;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LLMResultChunk {
    private Delta delta;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Delta {
        private Message message;
        private LLMUsage usage;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Message {
        private Object content;
        private java.util.List<AssistantPromptMessage.ToolCall> toolCalls;
    }
}
