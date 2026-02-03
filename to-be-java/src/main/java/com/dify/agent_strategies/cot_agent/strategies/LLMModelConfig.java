package com.dify.agent_strategies.cot_agent.strategies;

import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LLMModelConfig {
    private Map<String, Object> config;
}
