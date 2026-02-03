package com.dify.agent_strategies.cot_agent.strategies;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ReactChunk {
    private ReactState state;
    private String content;
}
