package com.dify.agent_strategies.cot_agent.strategies;

import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AssistantPromptMessage extends PromptMessage {
    private List<ToolCall> toolCalls;

    public AssistantPromptMessage(String content) {
        super(content);
    }

    public AssistantPromptMessage(String content, List<ToolCall> toolCalls) {
        super(content);
        this.toolCalls = toolCalls;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ToolCall {
        private String id;
        private String type;
        private ToolCallFunction function;

        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class ToolCallFunction {
            private String name;
            private String arguments;
        }
    }
}
