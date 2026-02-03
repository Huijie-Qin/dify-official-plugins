package com.dify.agent_strategies.cot_agent.strategies;

import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ToolInvokeMeta {
    private String error;

    public static ToolInvokeMeta errorInstance(String message) {
        return new ToolInvokeMeta(message);
    }

    public Map<String, Object> toDict() {
        return Map.of("error", error);
    }
}
