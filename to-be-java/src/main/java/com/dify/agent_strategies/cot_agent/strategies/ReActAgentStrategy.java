package com.dify.agent_strategies.cot_agent.strategies;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

public class ReActAgentStrategy extends AgentStrategy {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final List<String> ignoreObservationProviders = Arrays.asList("wenxin");

    private String query = "";
    private String instruction = "";
    private List<PromptMessage> historyPromptMessages = new ArrayList<>();
    private List<ToolEntity> promptMessagesTools = new ArrayList<>();

    public UserPromptMessage getUserPromptMessage() {
        return new UserPromptMessage(query);
    }

    public SystemPromptMessage getSystemPromptMessage() {
        AgentPromptEntity promptEntity = AgentPromptEntity.builder()
            .firstPrompt(((Map<String, Object>) ((Map<String, Object>) REACT_PROMPT_TEMPLATES.get("english"))
                .get("chat")).get("prompt").toString())
            .nextIteration(((Map<String, Object>) ((Map<String, Object>) REACT_PROMPT_TEMPLATES.get("english"))
                .get("chat")).get("agent_scratchpad").toString())
            .build();
        if (promptEntity == null) {
            throw new IllegalArgumentException("Agent prompt configuration is not set");
        }
        String systemPrompt = promptEntity.getFirstPrompt()
            .replace("{{instruction}}", instruction)
            .replace(
                "{{tools}}",
                toJson(
                    promptMessagesTools.stream()
                        .map(tool -> tool.modelDump("json"))
                        .toList()
                )
            )
            .replace(
                "{{tool_names}}",
                String.join(", ", promptMessagesTools.stream().map(ToolEntity::getName).toList())
            );
        return new SystemPromptMessage(systemPrompt);
    }

    public Iterable<AgentInvokeMessage> invoke(Map<String, Object> parameters) {
        ReActParams reactParams;
        try {
            reactParams = OBJECT_MAPPER.convertValue(parameters, ReActParams.class);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid parameters: " + e.getMessage(), e);
        }

        query = reactParams.getQuery();
        instruction = reactParams.getInstruction();
        List<AgentScratchpadUnit> agentScratchpad = new ArrayList<>();
        int iterationStep = 1;
        int maxIterationSteps = reactParams.getMaximumIterations();
        boolean runAgentState = true;
        Map<String, Optional<LLMUsage>> llmUsage = new HashMap<>();
        llmUsage.put("usage", Optional.empty());
        String finalAnswer = "";
        List<PromptMessage> promptMessages = new ArrayList<>();

        AgentModelConfig model = reactParams.getModel();
        List<String> stop = model.getCompletionParams() != null
            ? new ArrayList<>(model.getCompletionParams().getOrDefault("stop", Collections.emptyList()))
            : new ArrayList<>();
        if (!stop.contains("Observation") && !ignoreObservationProviders.contains(model.getProvider())) {
            stop.add("Observation");
        }

        historyPromptMessages = model.getHistoryPromptMessages();

        List<ToolEntity> tools = reactParams.getTools();
        Map<String, ToolEntity> toolInstances = tools != null
            ? tools.stream().collect(HashMap::new, (map, tool) -> map.put(tool.getIdentity().getName(), tool), HashMap::putAll)
            : new HashMap<>();
        if (reactParams.getModel().getCompletionParams() == null) {
            reactParams.getModel().setCompletionParams(new HashMap<>());
        }
        List<ToolEntity> promptMessagesTools = initPromptTools(tools);
        this.promptMessagesTools = promptMessagesTools;

        List<AgentInvokeMessage> outputs = new ArrayList<>();

        while (runAgentState && iterationStep <= maxIterationSteps) {
            runAgentState = false;
            double roundStartedAt = nowSeconds();
            ToolInvokeMessage.LogMessage roundLog = createLogMessage(
                "ROUND " + iterationStep,
                new HashMap<>(),
                Map.of(LogMetadata.startedAt, roundStartedAt),
                ToolInvokeMessage.LogMessage.LogStatus.START
            );
            outputs.add(roundLog);
            List<String> messageFileIds = new ArrayList<>();

            promptMessages = organizePromptMessages(agentScratchpad, query);
            if (model.getEntity() != null && model.getCompletionParams() != null) {
                recalcLlmMaxTokens(model.getEntity(), promptMessages, model.getCompletionParams());
            }

            Iterable<?> chunks = session.getModel().getLlm().invoke(
                new LLMModelConfig(model.modelDump("json")),
                promptMessages,
                true,
                stop
            );

            Map<String, Optional<LLMUsage>> usageDict = new HashMap<>();
            usageDict.put("usage", Optional.empty());
            Iterable<Object> reactChunks = CotAgentOutputParser.handleReactStreamOutput(chunks, usageDict);

            AgentScratchpadUnit scratchpad = new AgentScratchpadUnit("", "", "", "", null);

            double modelStartedAt = nowSeconds();
            ToolInvokeMessage.LogMessage modelLog = createLogMessage(
                model.getModel() + " Thought",
                new HashMap<>(),
                Map.of(
                    LogMetadata.startedAt, modelStartedAt,
                    LogMetadata.provider, model.getProvider()
                ),
                roundLog,
                ToolInvokeMessage.LogMessage.LogStatus.START
            );
            outputs.add(modelLog);

            for (Object reactChunk : reactChunks) {
                if (reactChunk instanceof AgentScratchpadUnit.Action action) {
                    scratchpad.setAgentResponse(Optional.ofNullable(scratchpad.getAgentResponse()).orElse("")
                        + toJson(action.modelDump()));
                    scratchpad.setActionStr(toJson(action.modelDump()));
                    scratchpad.setAction(action);
                } else if (reactChunk instanceof ReactChunk chunk) {
                    ReactState chunkState = chunk.getState();
                    String content = chunk.getContent();
                    outputs.add(createTextMessage(content));
                    if (chunkState == ReactState.ANSWER) {
                        finalAnswer += content;
                    } else if (chunkState == ReactState.THINKING) {
                        scratchpad.setAgentResponse(Optional.ofNullable(scratchpad.getAgentResponse()).orElse("") + content);
                        scratchpad.setThought(Optional.ofNullable(scratchpad.getThought()).orElse("") + content);
                    }
                }
            }

            scratchpad.setThought(
                Optional.ofNullable(scratchpad.getThought()).map(String::trim)
                    .filter(thought -> !thought.isBlank())
                    .orElse("I am thinking about how to help you")
            );
            agentScratchpad.add(scratchpad);

            Optional<LLMUsage> usage = usageDict.getOrDefault("usage", Optional.empty());
            usage.ifPresent(value -> increaseUsage(llmUsage, value));
            usageDict.putIfAbsent("usage", Optional.of(LLMUsage.emptyUsage()));

            Map<String, Object> actionDict = scratchpad.getAction() != null
                ? scratchpad.getAction().toDict()
                : Map.of("action", scratchpad.getAgentResponse());

            outputs.add(finishLogMessage(
                modelLog,
                mergeMaps(Map.of("thought", scratchpad.getThought()), actionDict),
                Map.of(
                    LogMetadata.startedAt, modelStartedAt,
                    LogMetadata.finishedAt, nowSeconds(),
                    LogMetadata.elapsedTime, nowSeconds() - modelStartedAt,
                    LogMetadata.provider, model.getProvider(),
                    LogMetadata.totalPrice, usage.map(LLMUsage::getTotalPrice).orElse(0.0),
                    LogMetadata.currency, usage.map(LLMUsage::getCurrency).orElse(""),
                    LogMetadata.totalTokens, usage.map(LLMUsage::getTotalTokens).orElse(0)
                )
            ));

            if (scratchpad.getAction() == null) {
                finalAnswer = scratchpad.getThought();
            } else if ("final answer".equalsIgnoreCase(scratchpad.getAction().getActionName())) {
                Object actionInput = scratchpad.getAction().getActionInput();
                if (actionInput instanceof Map) {
                    finalAnswer = toJson(actionInput);
                } else if (actionInput instanceof String) {
                    finalAnswer = (String) actionInput;
                } else {
                    finalAnswer = String.valueOf(actionInput);
                }
            } else {
                if (iterationStep == maxIterationSteps) {
                    String toolName = scratchpad.getAction().getActionName();
                    double toolCallStartedAt = nowSeconds();
                    ToolInvokeMessage.LogMessage toolCallLog = createLogMessage(
                        "CALL " + toolName,
                        new HashMap<>(),
                        Map.of(
                            LogMetadata.startedAt, toolCallStartedAt,
                            LogMetadata.provider, toolInstances.containsKey(toolName)
                                ? toolInstances.get(toolName).getIdentity().getProvider()
                                : ""
                        ),
                        roundLog,
                        ToolInvokeMessage.LogMessage.LogStatus.START
                    );
                    outputs.add(toolCallLog);

                    String errorMessage = "Maximum iteration limit (" + maxIterationSteps + ") reached. "
                        + "Cannot call tool '" + toolName + "'. "
                        + "Please consider increasing the iteration limit.";
                    scratchpad.setObservation(errorMessage);
                    scratchpad.setAgentResponse(errorMessage);
                    finalAnswer = errorMessage;

                    outputs.add(finishLogMessage(
                        toolCallLog,
                        Map.of(
                            "tool_name", toolName,
                            "tool_call_args", scratchpad.getAction().getActionInput(),
                            "output", errorMessage
                        ),
                        Map.of(
                            LogMetadata.startedAt, toolCallStartedAt,
                            LogMetadata.provider, toolInstances.containsKey(toolName)
                                ? toolInstances.get(toolName).getIdentity().getProvider()
                                : "",
                            LogMetadata.finishedAt, nowSeconds(),
                            LogMetadata.elapsedTime, nowSeconds() - toolCallStartedAt
                        )
                    ));
                } else {
                    runAgentState = true;
                    double toolCallStartedAt = nowSeconds();
                    String toolName = scratchpad.getAction().getActionName();
                    ToolInvokeMessage.LogMessage toolCallLog = createLogMessage(
                        "CALL " + toolName,
                        new HashMap<>(),
                        Map.of(
                            LogMetadata.startedAt, nowSeconds(),
                            LogMetadata.provider, toolInstances.containsKey(toolName)
                                ? toolInstances.get(toolName).getIdentity().getProvider()
                                : ""
                        ),
                        roundLog,
                        ToolInvokeMessage.LogMessage.LogStatus.START
                    );
                    outputs.add(toolCallLog);
                    ToolInvokeOutcome outcome = handleInvokeAction(
                        scratchpad.getAction(),
                        toolInstances,
                        messageFileIds
                    );
                    scratchpad.setObservation(outcome.getResult());
                    scratchpad.setAgentResponse(outcome.getResult());
                    outputs.addAll(outcome.getAdditionalMessages());
                    outputs.add(finishLogMessage(
                        toolCallLog,
                        Map.of(
                            "tool_name", toolName,
                            "tool_call_args", outcome.getParameters(),
                            "output", outcome.getResult()
                        ),
                        Map.of(
                            LogMetadata.startedAt, toolCallStartedAt,
                            LogMetadata.provider, toolInstances.containsKey(toolName)
                                ? toolInstances.get(toolName).getIdentity().getProvider()
                                : "",
                            LogMetadata.finishedAt, nowSeconds(),
                            LogMetadata.elapsedTime, nowSeconds() - toolCallStartedAt
                        )
                    ));
                }

                for (ToolEntity promptTool : promptMessagesTools) {
                    updatePromptMessageTool(toolInstances.get(promptTool.getName()), promptTool);
                }
            }

            outputs.add(finishLogMessage(
                roundLog,
                Map.of(
                    "action_name", scratchpad.getAction() != null ? scratchpad.getAction().getActionName() : "",
                    "action_input", scratchpad.getAction() != null ? scratchpad.getAction().getActionInput() : "",
                    "thought", scratchpad.getThought(),
                    "observation", scratchpad.getObservation()
                ),
                Map.of(
                    LogMetadata.startedAt, roundStartedAt,
                    LogMetadata.finishedAt, nowSeconds(),
                    LogMetadata.elapsedTime, nowSeconds() - roundStartedAt,
                    LogMetadata.totalPrice, usage.map(LLMUsage::getTotalPrice).orElse(0.0),
                    LogMetadata.currency, usage.map(LLMUsage::getCurrency).orElse(""),
                    LogMetadata.totalTokens, usage.map(LLMUsage::getTotalTokens).orElse(0)
                )
            ));
            iterationStep += 1;
        }

        if (reactParams.getContext() != null) {
            outputs.add(createRetrieverResourceMessage(
                reactParams.getContext().stream()
                    .map(ctx -> new ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource(
                        ctx.getContent(),
                        ctx.getMetadata().get("position"),
                        ctx.getMetadata().get("dataset_id"),
                        ctx.getMetadata().get("dataset_name"),
                        ctx.getMetadata().get("document_id"),
                        ctx.getMetadata().get("document_name"),
                        ctx.getMetadata().get("document_data_source_type"),
                        ctx.getMetadata().get("segment_id"),
                        ctx.getMetadata().get("retriever_from"),
                        ctx.getMetadata().get("score"),
                        ctx.getMetadata().get("segment_hit_count"),
                        ctx.getMetadata().get("segment_word_count"),
                        ctx.getMetadata().get("segment_position"),
                        ctx.getMetadata().get("segment_index_node_hash"),
                        ctx.getMetadata().get("page"),
                        ctx.getMetadata().get("doc_metadata")
                    ))
                    .toList(),
                ""
            ));
        }

        Optional<LLMUsage> usage = llmUsage.getOrDefault("usage", Optional.empty());
        outputs.add(createJsonMessage(
            Map.of(
                "execution_metadata",
                Map.of(
                    LogMetadata.totalPrice, usage.map(LLMUsage::getTotalPrice).orElse(0.0),
                    LogMetadata.currency, usage.map(LLMUsage::getCurrency).orElse(""),
                    LogMetadata.totalTokens, usage.map(LLMUsage::getTotalTokens).orElse(0)
                )
            )
        ));

        return outputs;
    }

    private List<PromptMessage> organizeUserQuery(String query, List<PromptMessage> promptMessages) {
        promptMessages.add(new UserPromptMessage(query));
        return promptMessages;
    }

    private List<PromptMessage> organizePromptMessages(List<AgentScratchpadUnit> agentScratchpad, String query) {
        SystemPromptMessage systemMessage = getSystemPromptMessage();

        List<PromptMessage> assistantMessages = new ArrayList<>();
        if (!agentScratchpad.isEmpty()) {
            AssistantPromptMessage assistantMessage = new AssistantPromptMessage("");
            for (AgentScratchpadUnit unit : agentScratchpad) {
                if (unit.isFinal()) {
                    assistantMessage.setContent(assistantMessage.getContent()
                        + "Final Answer: " + unit.getAgentResponse());
                } else {
                    assistantMessage.setContent(assistantMessage.getContent()
                        + "Thought: " + unit.getThought() + "\n\n");
                    if (unit.getActionStr() != null && !unit.getActionStr().isEmpty()) {
                        assistantMessage.setContent(assistantMessage.getContent()
                            + "Action: " + unit.getActionStr() + "\n\n");
                    }
                    if (unit.getObservation() != null && !unit.getObservation().isEmpty()) {
                        assistantMessage.setContent(assistantMessage.getContent()
                            + "Observation: " + unit.getObservation() + "\n\n");
                    }
                }
            }
            assistantMessages.add(assistantMessage);
        }

        List<PromptMessage> queryMessages = organizeUserQuery(query, new ArrayList<>());

        List<PromptMessage> messages = new ArrayList<>();
        messages.add(systemMessage);
        messages.addAll(historyPromptMessages);
        messages.addAll(queryMessages);
        if (!assistantMessages.isEmpty()) {
            messages.addAll(assistantMessages);
            messages.add(new UserPromptMessage("continue"));
        }

        return messages;
    }

    private ToolInvokeOutcome handleInvokeAction(
        AgentScratchpadUnit.Action action,
        Map<String, ToolEntity> toolInstances,
        List<String> messageFileIds
    ) {
        String toolCallName = action.getActionName();
        Object toolCallArgs = action.getActionInput();
        ToolEntity toolInstance = toolInstances.get(toolCallName);

        if (toolInstance == null) {
            return new ToolInvokeOutcome(
                "there is not a tool named " + toolCallName,
                toolCallArgs,
                new ArrayList<>()
            );
        }

        if (toolCallArgs instanceof String) {
            try {
                toolCallArgs = OBJECT_MAPPER.readValue((String) toolCallArgs, Map.class);
            } catch (JsonProcessingException e) {
                List<String> params = toolInstance.getParameters().stream()
                    .filter(param -> param.getForm() == ToolParameter.ToolParameterForm.LLM)
                    .map(ToolParameter::getName)
                    .toList();
                if (params.size() > 1) {
                    throw new IllegalArgumentException("tool call args is not a valid json string", e);
                }
                toolCallArgs = params.size() == 1
                    ? Map.of(params.get(0), toolCallArgs)
                    : new HashMap<>();
            }
        }

        Map<String, Object> toolInvokeParameters = new HashMap<>(toolInstance.getRuntimeParameters());
        toolInvokeParameters.putAll((Map<String, Object>) toolCallArgs);
        String result;
        List<ToolInvokeMessage> additionalMessages = new ArrayList<>();
        try {
            Iterable<ToolInvokeMessage> toolInvokeResponses = session.getTool().invoke(
                ToolProviderType.valueOf(toolInstance.getProviderType()),
                toolInstance.getIdentity().getProvider(),
                toolInstance.getIdentity().getName(),
                toolInvokeParameters
            );
            StringBuilder resultBuilder = new StringBuilder();
            for (ToolInvokeMessage response : toolInvokeResponses) {
                if (response.getType() == ToolInvokeMessage.MessageType.TEXT) {
                    resultBuilder.append(((ToolInvokeMessage.TextMessage) response.getMessage()).getText());
                } else if (response.getType() == ToolInvokeMessage.MessageType.LINK) {
                    resultBuilder.append("result link: ")
                        .append(((ToolInvokeMessage.TextMessage) response.getMessage()).getText())
                        .append(". please tell user to check it.");
                } else if (response.getType() == ToolInvokeMessage.MessageType.IMAGE_LINK
                    || response.getType() == ToolInvokeMessage.MessageType.IMAGE) {
                    additionalMessages.add(response);
                    String imageLinkText = ((ToolInvokeMessage.TextMessage) response.getMessage()).getText();
                    resultBuilder.append("Image has been successfully generated and saved to: ")
                        .append(imageLinkText)
                        .append(". The image file is now available for download. ")
                        .append("Please inform the user that the image has been created successfully.");
                } else if (response.getType() == ToolInvokeMessage.MessageType.JSON) {
                    String text = toJson(((ToolInvokeMessage.JsonMessage) response.getMessage()).getJsonObject());
                    resultBuilder.append("tool response: ").append(text).append('.');
                } else if (response.getType() == ToolInvokeMessage.MessageType.BLOB) {
                    resultBuilder.append("Generated file with ... ");
                    additionalMessages.add(response);
                } else {
                    resultBuilder.append("tool response: ").append(response.getMessage()).append('.');
                }
            }
            result = resultBuilder.toString();
        } catch (Exception e) {
            result = "tool invoke error: " + e.getMessage();
            additionalMessages = new ArrayList<>();
        }

        return new ToolInvokeOutcome(result, toolInvokeParameters, additionalMessages);
    }

    private AgentScratchpadUnit.Action convertDictToAction(Map<String, Object> action) {
        return new AgentScratchpadUnit.Action(
            (String) action.get("action"),
            action.get("action_input")
        );
    }

    private String formatAssistantMessage(List<AgentScratchpadUnit> agentScratchpad) {
        StringBuilder message = new StringBuilder();
        for (AgentScratchpadUnit scratchpad : agentScratchpad) {
            if (scratchpad.isFinal()) {
                message.append("Final Answer: ").append(scratchpad.getAgentResponse());
            } else {
                message.append("Thought: ").append(scratchpad.getThought()).append("\n\n");
                if (scratchpad.getActionStr() != null && !scratchpad.getActionStr().isEmpty()) {
                    message.append("Action: ").append(scratchpad.getActionStr()).append("\n\n");
                }
                if (scratchpad.getObservation() != null && !scratchpad.getObservation().isEmpty()) {
                    message.append("Observation: ").append(scratchpad.getObservation()).append("\n\n");
                }
            }
        }
        return message.toString();
    }

    private double nowSeconds() {
        return System.nanoTime() / 1_000_000_000.0;
    }

    private String toJson(Object value) {
        try {
            return OBJECT_MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize to JSON", e);
        }
    }

    private Map<String, Object> mergeMaps(Map<String, Object> left, Map<String, Object> right) {
        Map<String, Object> merged = new HashMap<>(left);
        merged.putAll(right);
        return merged;
    }

    @Data
    public static class LogMetadata {
        public static final String startedAt = "started_at";
        public static final String provider = "provider";
        public static final String finishedAt = "finished_at";
        public static final String elapsedTime = "elapsed_time";
        public static final String totalPrice = "total_price";
        public static final String currency = "currency";
        public static final String totalTokens = "total_tokens";
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ContextItem {
        private String content;
        private String title;
        private Map<String, Object> metadata;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ReActParams {
        private String query;
        private String instruction;
        private AgentModelConfig model;
        private List<ToolEntity> tools;
        @Builder.Default
        private int maximumIterations = 3;
        private List<ContextItem> context;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AgentPromptEntity {
        private String firstPrompt;
        private String nextIteration;
    }

    @Data
    @AllArgsConstructor
    public static class ToolInvokeOutcome {
        private String result;
        private Object parameters;
        private List<ToolInvokeMessage> additionalMessages;
    }
}
