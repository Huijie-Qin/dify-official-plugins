package com.dify.agent_strategies.cot_agent.strategies;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ToolPromptMessage extends PromptMessage {
    private String toolCallId;
    private String name;

    public ToolPromptMessage(String content, String toolCallId, String name) {
        super(content);
        this.toolCallId = toolCallId;
        this.name = name;
    }
}
