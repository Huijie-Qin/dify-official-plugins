package com.dify.agent_strategies.cot_agent.strategies;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AgentModelConfig {
    private String model;
    private String provider;
    private Map<String, Object> completionParams = new HashMap<>();
    private List<PromptMessage> historyPromptMessages = new ArrayList<>();
    private ModelEntity entity;

    public Map<String, Object> modelDump(String mode) {
        Map<String, Object> dump = new HashMap<>();
        dump.put("model", model);
        dump.put("provider", provider);
        dump.put("completion_params", completionParams);
        dump.put("history_prompt_messages", historyPromptMessages);
        dump.put("entity", entity);
        return dump;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ModelEntity {
        private List<ModelFeature> features = new ArrayList<>();
    }
}
