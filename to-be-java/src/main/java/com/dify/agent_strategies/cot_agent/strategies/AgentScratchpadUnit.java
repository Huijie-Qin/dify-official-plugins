package com.dify.agent_strategies.cot_agent.strategies;

import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AgentScratchpadUnit {
    private String agentResponse;
    private String thought;
    private String actionStr;
    private String observation;
    private Action action;

    public boolean isFinal() {
        return action != null && "final answer".equalsIgnoreCase(action.getActionName());
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Action {
        private String actionName;
        private Object actionInput;

        public Map<String, Object> toDict() {
            return Map.of(
                "action", actionName,
                "action_input", actionInput
            );
        }

        public Map<String, Object> modelDump() {
            return toDict();
        }
    }
}
