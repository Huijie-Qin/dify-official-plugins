package com.dify.agent_strategies.cot_agent.strategies;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ToolParameter {
    private String name;
    private ToolParameterForm form;

    public enum ToolParameterForm {
        LLM,
        HUMAN
    }
}
