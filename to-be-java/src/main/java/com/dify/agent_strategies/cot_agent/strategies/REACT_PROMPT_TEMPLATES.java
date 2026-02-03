package com.dify.agent_strategies.cot_agent.strategies;

import java.util.Map;

public class REACT_PROMPT_TEMPLATES {
    private static final Map<String, Object> ENGLISH_CHAT = Map.of(
        "prompt", "{{instruction}}\n{{tools}}",
        "agent_scratchpad", ""
    );

    private static final Map<String, Object> ROOT = Map.of(
        "english", Map.of("chat", ENGLISH_CHAT)
    );

    public static Object get(String key) {
        return ROOT.get(key);
    }
}
