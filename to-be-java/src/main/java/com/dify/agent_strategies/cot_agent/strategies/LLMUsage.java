package com.dify.agent_strategies.cot_agent.strategies;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class LLMUsage {
    private double totalPrice;
    private String currency;
    private int totalTokens;
    private int promptTokens;
    private double promptUnitPrice;
    private double promptPriceUnit;
    private double promptPrice;
    private int completionTokens;
    private double completionUnitPrice;
    private double completionPriceUnit;
    private double completionPrice;
    private double latency;

    public static LLMUsage emptyUsage() {
        return new LLMUsage(0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
}
