package com.dify.agent_strategies.cot_agent.strategies;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ToolEntity {
    private String name;
    private Identity identity;
    private String providerType;
    private Map<String, Object> runtimeParameters = new HashMap<>();
    private List<ToolParameter> parameters;

    public Map<String, Object> modelDump(String mode) {
        Map<String, Object> dump = new HashMap<>();
        dump.put("name", name);
        dump.put("identity", identity);
        dump.put("provider_type", providerType);
        dump.put("runtime_parameters", runtimeParameters);
        dump.put("parameters", parameters);
        return dump;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Identity {
        private String name;
        private String provider;
    }
}
