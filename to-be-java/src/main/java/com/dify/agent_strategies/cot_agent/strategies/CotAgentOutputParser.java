package com.dify.agent_strategies.cot_agent.strategies;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class CotAgentOutputParser {
    public static Iterable<Object> handleReactStreamOutput(
        Iterable<?> chunks,
        Map<String, Optional<LLMUsage>> usageDict
    ) {
        return Collections.emptyList();
    }
}
