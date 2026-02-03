package com.dify.agent_strategies.cot_agent.strategies;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

public class FunctionCallingAgentStrategy extends AgentStrategy {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private String query = "";
    private String instruction = "";

    public UserPromptMessage getUserPromptMessage() {
        return new UserPromptMessage(query);
    }

    public SystemPromptMessage getSystemPromptMessage() {
        return new SystemPromptMessage(instruction);
    }

    public Iterable<AgentInvokeMessage> invoke(Map<String, Object> parameters) {
        FunctionCallingParams fcParams = OBJECT_MAPPER.convertValue(parameters, FunctionCallingParams.class);

        String query = fcParams.getQuery();
        this.query = query;
        instruction = fcParams.getInstruction();
        List<PromptMessage> historyPromptMessages = fcParams.getModel().getHistoryPromptMessages();
        historyPromptMessages.add(0, getSystemPromptMessage());
        historyPromptMessages.add(getUserPromptMessage());

        List<ToolEntity> tools = fcParams.getTools();
        Map<String, ToolEntity> toolInstances = tools != null
            ? tools.stream().collect(HashMap::new, (map, tool) -> map.put(tool.getIdentity().getName(), tool), HashMap::putAll)
            : new HashMap<>();
        List<ToolEntity> promptMessagesTools = initPromptTools(tools);

        boolean stream = fcParams.getModel().getEntity() != null
            && fcParams.getModel().getEntity().getFeatures() != null
            && fcParams.getModel().getEntity().getFeatures().contains(ModelFeature.STREAM_TOOL_CALL);
        AgentModelConfig model = fcParams.getModel();
        List<String> stop = model.getCompletionParams() != null
            ? model.getCompletionParams().getOrDefault("stop", new ArrayList<>())
            : new ArrayList<>();

        int iterationStep = 1;
        int maxIterationSteps = fcParams.getMaximumIterations();
        List<PromptMessage> currentThoughts = new ArrayList<>();
        boolean functionCallState = true;
        Map<String, Optional<LLMUsage>> llmUsage = new HashMap<>();
        llmUsage.put("usage", Optional.empty());
        String finalAnswer = "";

        List<AgentInvokeMessage> outputs = new ArrayList<>();

        while (functionCallState && iterationStep <= maxIterationSteps) {
            functionCallState = false;
            double roundStartedAt = nowSeconds();
            ToolInvokeMessage.LogMessage roundLog = createLogMessage(
                "ROUND " + iterationStep,
                new HashMap<>(),
                Map.of(LogMetadata.startedAt, roundStartedAt),
                ToolInvokeMessage.LogMessage.LogStatus.START
            );
            outputs.add(roundLog);

            List<PromptMessage> promptMessages = organizePromptMessages(
                currentThoughts,
                historyPromptMessages,
                model
            );
            if (model.getEntity() != null && model.getCompletionParams() != null) {
                recalcLlmMaxTokens(model.getEntity(), promptMessages, model.getCompletionParams());
            }

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

            LLMModelConfig modelConfig = new LLMModelConfig(model.modelDump("json"));
            Object chunks = session.getModel().getLlm().invoke(
                modelConfig,
                promptMessages,
                stop,
                stream,
                promptMessagesTools
            );

            List<ToolCall> toolCalls = new ArrayList<>();
            StringBuilder response = new StringBuilder();
            String toolCallNames = "";
            LLMUsage currentLlmUsage = null;

            if (chunks instanceof Iterable<?>) {
                for (Object chunkObj : (Iterable<?>) chunks) {
                    LLMResultChunk chunk = (LLMResultChunk) chunkObj;
                    if (checkToolCalls(chunk)) {
                        functionCallState = true;
                        toolCalls.addAll(extractToolCalls(chunk));
                        toolCallNames = String.join(
                            ";",
                            toolCalls.stream().map(ToolCall::getToolCallName).toList()
                        );
                    }

                    if (chunk.getDelta().getMessage() != null && chunk.getDelta().getMessage().getContent() != null) {
                        Object content = chunk.getDelta().getMessage().getContent();
                        if (content instanceof List<?> contentList) {
                            for (Object contentItem : contentList) {
                                PromptMessageContent contentMessage = (PromptMessageContent) contentItem;
                                response.append(contentMessage.getData());
                                if (!functionCallState || iterationStep == maxIterationSteps) {
                                    outputs.add(createTextMessage(contentMessage.getData()));
                                }
                            }
                        } else {
                            response.append(content.toString());
                            if (!functionCallState || iterationStep == maxIterationSteps) {
                                outputs.add(createTextMessage(content.toString()));
                            }
                        }
                    }

                    if (chunk.getDelta().getUsage() != null) {
                        increaseUsage(llmUsage, chunk.getDelta().getUsage());
                        currentLlmUsage = chunk.getDelta().getUsage();
                    }
                }
            } else {
                LLMResult result = (LLMResult) chunks;
                if (checkBlockingToolCalls(result)) {
                    functionCallState = true;
                    toolCalls.addAll(extractBlockingToolCalls(result));
                    toolCallNames = String.join(
                        ";",
                        toolCalls.stream().map(ToolCall::getToolCallName).toList()
                    );
                }

                if (result.getUsage() != null) {
                    increaseUsage(llmUsage, result.getUsage());
                    currentLlmUsage = result.getUsage();
                }

                if (result.getMessage() != null && result.getMessage().getContent() != null) {
                    Object content = result.getMessage().getContent();
                    if (content instanceof List<?> contentList) {
                        for (Object contentItem : contentList) {
                            response.append(((PromptMessageContent) contentItem).getData());
                        }
                    } else {
                        response.append(content.toString());
                    }
                }

                if (result.getMessage() != null && result.getMessage().getContent() == null) {
                    result.getMessage().setContent("");
                }

                if (result.getMessage() != null) {
                    Object content = result.getMessage().getContent();
                    if (content instanceof String) {
                        outputs.add(createTextMessage(content.toString()));
                    } else if (content instanceof List<?> contentList) {
                        for (Object contentItem : contentList) {
                            outputs.add(createTextMessage(((PromptMessageContent) contentItem).getData()));
                        }
                    }
                }
            }

            outputs.add(finishLogMessage(
                modelLog,
                Map.of(
                    "output", response.toString(),
                    "tool_name", toolCallNames,
                    "tool_input", toolCalls.stream()
                        .map(call -> Map.of("name", call.getToolCallName(), "args", call.getToolCallArgs()))
                        .toList()
                ),
                Map.of(
                    LogMetadata.startedAt, modelStartedAt,
                    LogMetadata.finishedAt, nowSeconds(),
                    LogMetadata.elapsedTime, nowSeconds() - modelStartedAt,
                    LogMetadata.provider, model.getProvider(),
                    LogMetadata.totalPrice, currentLlmUsage != null ? currentLlmUsage.getTotalPrice() : 0.0,
                    LogMetadata.currency, currentLlmUsage != null ? currentLlmUsage.getCurrency() : "",
                    LogMetadata.totalTokens, currentLlmUsage != null ? currentLlmUsage.getTotalTokens() : 0
                )
            ));

            if (!toolCalls.isEmpty()) {
                List<AssistantPromptMessage.ToolCall> toolCallObjects = toolCalls.stream()
                    .map(call -> new AssistantPromptMessage.ToolCall(
                        call.getToolCallId(),
                        "function",
                        new AssistantPromptMessage.ToolCall.ToolCallFunction(
                            call.getToolCallName(),
                            toJson(call.getToolCallArgs())
                        )
                    ))
                    .toList();
                currentThoughts.add(new AssistantPromptMessage(response.toString(), toolCallObjects));
            } else if (!response.toString().trim().isEmpty()) {
                currentThoughts.add(new AssistantPromptMessage(response.toString(), new ArrayList<>()));
            }

            finalAnswer += response + "\n";

            List<Map<String, Object>> toolResponses = new ArrayList<>();
            if (!toolCalls.isEmpty() && iterationStep == maxIterationSteps && maxIterationSteps > 1) {
                for (ToolCall toolCall : toolCalls) {
                    double toolCallStartedAt = nowSeconds();
                    ToolInvokeMessage.LogMessage toolCallLog = createLogMessage(
                        "CALL " + toolCall.getToolCallName(),
                        new HashMap<>(),
                        Map.of(
                            LogMetadata.startedAt, nowSeconds(),
                            LogMetadata.provider, toolInstances.containsKey(toolCall.getToolCallName())
                                ? toolInstances.get(toolCall.getToolCallName()).getIdentity().getProvider()
                                : ""
                        ),
                        roundLog,
                        ToolInvokeMessage.LogMessage.LogStatus.START
                    );
                    outputs.add(toolCallLog);

                    Map<String, Object> toolResponse = Map.of(
                        "tool_call_id", toolCall.getToolCallId(),
                        "tool_call_name", toolCall.getToolCallName(),
                        "tool_response", "Maximum iteration limit (" + maxIterationSteps + ") reached. "
                            + "Cannot call tool '" + toolCall.getToolCallName() + "'. "
                            + "Please consider increasing the iteration limit."
                    );
                    toolResponses.add(toolResponse);

                    outputs.add(finishLogMessage(
                        toolCallLog,
                        Map.of("output", toolResponse),
                        Map.of(
                            LogMetadata.startedAt, toolCallStartedAt,
                            LogMetadata.provider, toolInstances.containsKey(toolCall.getToolCallName())
                                ? toolInstances.get(toolCall.getToolCallName()).getIdentity().getProvider()
                                : "",
                            LogMetadata.finishedAt, nowSeconds(),
                            LogMetadata.elapsedTime, nowSeconds() - toolCallStartedAt
                        )
                    ));

                    currentThoughts.add(new AssistantPromptMessage(
                        "",
                        List.of(new AssistantPromptMessage.ToolCall(
                            toolCall.getToolCallId(),
                            "function",
                            new AssistantPromptMessage.ToolCall.ToolCallFunction(
                                toolCall.getToolCallName(),
                                toJson(toolCall.getToolCallArgs())
                            )
                        ))
                    ));
                    currentThoughts.add(new ToolPromptMessage(
                        toolResponse.get("tool_response").toString(),
                        toolCall.getToolCallId(),
                        toolCall.getToolCallName()
                    ));
                }
            } else {
                for (ToolCall toolCall : toolCalls) {
                    ToolEntity toolInstance = toolInstances.get(toolCall.getToolCallName());
                    double toolCallStartedAt = nowSeconds();
                    ToolInvokeMessage.LogMessage toolCallLog = createLogMessage(
                        "CALL " + toolCall.getToolCallName(),
                        new HashMap<>(),
                        Map.of(
                            LogMetadata.startedAt, nowSeconds(),
                            LogMetadata.provider, toolInstance.getIdentity().getProvider()
                        ),
                        roundLog,
                        ToolInvokeMessage.LogMessage.LogStatus.START
                    );
                    outputs.add(toolCallLog);
                    Map<String, Object> toolResponse;
                    if (toolInstance == null) {
                        toolResponse = Map.of(
                            "tool_call_id", toolCall.getToolCallId(),
                            "tool_call_name", toolCall.getToolCallName(),
                            "tool_response", "there is not a tool named " + toolCall.getToolCallName(),
                            "meta", ToolInvokeMeta.errorInstance(
                                "there is not a tool named " + toolCall.getToolCallName()
                            ).toDict()
                        );
                    } else {
                        String toolResult;
                        try {
                            Iterable<ToolInvokeMessage> toolInvokeResponses = session.getTool().invoke(
                                ToolProviderType.valueOf(toolInstance.getProviderType()),
                                toolInstance.getIdentity().getProvider(),
                                toolInstance.getIdentity().getName(),
                                mergeMaps(toolInstance.getRuntimeParameters(), toolCall.getToolCallArgs())
                            );
                            StringBuilder toolResultBuilder = new StringBuilder();
                            for (ToolInvokeMessage toolInvokeResponse : toolInvokeResponses) {
                                if (toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.TEXT) {
                                    toolResultBuilder.append(((ToolInvokeMessage.TextMessage) toolInvokeResponse.getMessage())
                                        .getText());
                                } else if (toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.LINK) {
                                    toolResultBuilder.append("result link: ")
                                        .append(((ToolInvokeMessage.TextMessage) toolInvokeResponse.getMessage()).getText())
                                        .append(". please tell user to check it.");
                                } else if (toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.IMAGE_LINK
                                    || toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.IMAGE) {
                                    if (toolInvokeResponse.getMessage() instanceof ToolInvokeMessage.TextMessage message
                                        && message.getText().startsWith("/files/")) {
                                        try {
                                            byte[] fileContent = readFileBytes(message.getText());
                                            outputs.add(createBlobMessage(
                                                fileContent,
                                                Map.of(
                                                    "mime_type", "image/png",
                                                    "filename", extractFileName(message.getText())
                                                )
                                            ));
                                        } catch (Exception e) {
                                            outputs.add(createTextMessage("Failed to create blob message: " + e));
                                        }
                                    }
                                    toolResultBuilder.append(
                                        "image has been created and sent to user already, "
                                            + "you do not need to create it, just tell the user to check it now."
                                    );
                                    outputs.add(toolInvokeResponse);
                                } else if (toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.JSON) {
                                    String text = toJson(((ToolInvokeMessage.JsonMessage) toolInvokeResponse.getMessage())
                                        .getJsonObject());
                                    toolResultBuilder.append("tool response: ").append(text).append('.');
                                } else if (toolInvokeResponse.getType() == ToolInvokeMessage.MessageType.BLOB) {
                                    toolResultBuilder.append("Generated file ... ");
                                    outputs.add(toolInvokeResponse);
                                } else {
                                    toolResultBuilder.append("tool response: ")
                                        .append(toolInvokeResponse.getMessage())
                                        .append('.');
                                }
                            }
                            toolResult = toolResultBuilder.toString();
                        } catch (Exception e) {
                            toolResult = "tool invoke error: " + e.getMessage();
                        }

                        toolResponse = Map.of(
                            "tool_call_id", toolCall.getToolCallId(),
                            "tool_call_name", toolCall.getToolCallName(),
                            "tool_call_input", mergeMaps(toolInstance.getRuntimeParameters(), toolCall.getToolCallArgs()),
                            "tool_response", toolResult
                        );
                    }

                    outputs.add(finishLogMessage(
                        toolCallLog,
                        Map.of("output", toolResponse),
                        Map.of(
                            LogMetadata.startedAt, toolCallStartedAt,
                            LogMetadata.provider, toolInstance.getIdentity().getProvider(),
                            LogMetadata.finishedAt, nowSeconds(),
                            LogMetadata.elapsedTime, nowSeconds() - toolCallStartedAt
                        )
                    ));
                    toolResponses.add(toolResponse);
                    if (toolResponse.get("tool_response") != null) {
                        currentThoughts.add(new ToolPromptMessage(
                            toolResponse.get("tool_response").toString(),
                            toolCall.getToolCallId(),
                            toolCall.getToolCallName()
                        ));
                    }
                }
            }

            if (!toolCalls.isEmpty()) {
                outputs.add(createTextMessage("\n"));
            }

            for (ToolEntity promptTool : promptMessagesTools) {
                updatePromptMessageTool(toolInstances.get(promptTool.getName()), promptTool);
            }

            outputs.add(finishLogMessage(
                roundLog,
                Map.of(
                    "output", Map.of(
                        "llm_response", response.toString(),
                        "tool_responses", toolResponses
                    )
                ),
                Map.of(
                    LogMetadata.startedAt, roundStartedAt,
                    LogMetadata.finishedAt, nowSeconds(),
                    LogMetadata.elapsedTime, nowSeconds() - roundStartedAt,
                    LogMetadata.totalPrice, currentLlmUsage != null ? currentLlmUsage.getTotalPrice() : 0.0,
                    LogMetadata.currency, currentLlmUsage != null ? currentLlmUsage.getCurrency() : "",
                    LogMetadata.totalTokens, currentLlmUsage != null ? currentLlmUsage.getTotalTokens() : 0
                )
            ));

            if (!toolResponses.isEmpty() && maxIterationSteps == 1) {
                for (Map<String, Object> resp : toolResponses) {
                    outputs.add(createTextMessage(resp.get("tool_response").toString()));
                }
            }

            iterationStep += 1;
        }

        if (fcParams.getContext() != null) {
            outputs.add(createRetrieverResourceMessage(
                fcParams.getContext().stream()
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

        ExecutionMetadata metadata = ExecutionMetadata.fromLlmUsage(llmUsage.get("usage").orElse(null));
        outputs.add(createJsonMessage(Map.of("execution_metadata", metadata.toMap())));
        return outputs;
    }

    public boolean checkToolCalls(LLMResultChunk llmResultChunk) {
        return llmResultChunk.getDelta().getMessage().getToolCalls() != null
            && !llmResultChunk.getDelta().getMessage().getToolCalls().isEmpty();
    }

    public boolean checkBlockingToolCalls(LLMResult llmResult) {
        return llmResult.getMessage().getToolCalls() != null
            && !llmResult.getMessage().getToolCalls().isEmpty();
    }

    public List<ToolCall> extractToolCalls(LLMResultChunk llmResultChunk) {
        List<ToolCall> toolCalls = new ArrayList<>();
        for (AssistantPromptMessage.ToolCall promptMessage : llmResultChunk.getDelta().getMessage().getToolCalls()) {
            Map<String, Object> args = new HashMap<>();
            if (!promptMessage.getFunction().getArguments().isEmpty()) {
                args = readJson(promptMessage.getFunction().getArguments());
            }
            toolCalls.add(new ToolCall(promptMessage.getId(), promptMessage.getFunction().getName(), args));
        }
        return toolCalls;
    }

    public List<ToolCall> extractBlockingToolCalls(LLMResult llmResult) {
        List<ToolCall> toolCalls = new ArrayList<>();
        for (AssistantPromptMessage.ToolCall promptMessage : llmResult.getMessage().getToolCalls()) {
            Map<String, Object> args = new HashMap<>();
            if (!promptMessage.getFunction().getArguments().isEmpty()) {
                args = readJson(promptMessage.getFunction().getArguments());
            }
            toolCalls.add(new ToolCall(promptMessage.getId(), promptMessage.getFunction().getName(), args));
        }
        return toolCalls;
    }

    public List<PromptMessage> initSystemMessage(String promptTemplate, List<PromptMessage> promptMessages) {
        if (promptMessages.isEmpty() && promptTemplate != null && !promptTemplate.isEmpty()) {
            return List.of(new SystemPromptMessage(promptTemplate));
        }

        if (!promptMessages.isEmpty()
            && !(promptMessages.get(0) instanceof SystemPromptMessage)
            && promptTemplate != null
            && !promptTemplate.isEmpty()) {
            promptMessages.add(0, new SystemPromptMessage(promptTemplate));
        }

        return promptMessages != null ? promptMessages : new ArrayList<>();
    }

    public List<PromptMessage> clearUserPromptImageMessages(List<PromptMessage> promptMessages) {
        List<PromptMessage> clearedMessages = new ArrayList<>();
        for (PromptMessage promptMessage : promptMessages) {
            if (promptMessage instanceof UserPromptMessage userPromptMessage
                && userPromptMessage.getContent() instanceof List<?> contentList) {
                List<String> parts = new ArrayList<>();
                for (Object contentItem : contentList) {
                    PromptMessageContent content = (PromptMessageContent) contentItem;
                    if (content.getType() == PromptMessageContentType.TEXT) {
                        parts.add(content.getData());
                    } else if (content.getType() == PromptMessageContentType.IMAGE) {
                        parts.add("[image]");
                    } else {
                        parts.add("[file]");
                    }
                }
                userPromptMessage.setContent(String.join("\n", parts));
            }
            clearedMessages.add(promptMessage);
        }
        return clearedMessages;
    }

    public List<PromptMessage> organizePromptMessages(
        List<PromptMessage> currentThoughts,
        List<PromptMessage> historyPromptMessages,
        AgentModelConfig model
    ) {
        List<PromptMessage> promptMessages = new ArrayList<>();
        promptMessages.addAll(historyPromptMessages);
        promptMessages.addAll(currentThoughts);

        boolean supportsVision = model != null
            && model.getEntity() != null
            && model.getEntity().getFeatures() != null
            && model.getEntity().getFeatures().contains(ModelFeature.VISION);

        if (!supportsVision || !currentThoughts.isEmpty()) {
            promptMessages = clearUserPromptImageMessages(promptMessages);
        }

        return promptMessages;
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

    private Map<String, Object> readJson(String value) {
        try {
            return OBJECT_MAPPER.readValue(value, Map.class);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Invalid JSON string", e);
        }
    }

    private Map<String, Object> mergeMaps(Map<String, Object> left, Map<String, Object> right) {
        Map<String, Object> merged = new HashMap<>(left);
        merged.putAll(right);
        return merged;
    }

    private byte[] readFileBytes(String path) throws Exception {
        return Files.readAllBytes(Path.of(path));
    }

    private String extractFileName(String path) {
        int lastSlash = path.lastIndexOf('/');
        return lastSlash >= 0 ? path.substring(lastSlash + 1) : path;
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
    public static class ExecutionMetadata {
        @Builder.Default
        private double totalPrice = 0.0;
        @Builder.Default
        private String currency = "";
        @Builder.Default
        private int totalTokens = 0;
        @Builder.Default
        private int promptTokens = 0;
        @Builder.Default
        private double promptUnitPrice = 0.0;
        @Builder.Default
        private double promptPriceUnit = 0.0;
        @Builder.Default
        private double promptPrice = 0.0;
        @Builder.Default
        private int completionTokens = 0;
        @Builder.Default
        private double completionUnitPrice = 0.0;
        @Builder.Default
        private double completionPriceUnit = 0.0;
        @Builder.Default
        private double completionPrice = 0.0;
        @Builder.Default
        private double latency = 0.0;

        public static ExecutionMetadata fromLlmUsage(LLMUsage usage) {
            if (usage == null) {
                return ExecutionMetadata.builder().build();
            }

            return ExecutionMetadata.builder()
                .totalPrice(usage.getTotalPrice())
                .currency(usage.getCurrency())
                .totalTokens(usage.getTotalTokens())
                .promptTokens(usage.getPromptTokens())
                .promptUnitPrice(usage.getPromptUnitPrice())
                .promptPriceUnit(usage.getPromptPriceUnit())
                .promptPrice(usage.getPromptPrice())
                .completionTokens(usage.getCompletionTokens())
                .completionUnitPrice(usage.getCompletionUnitPrice())
                .completionPriceUnit(usage.getCompletionPriceUnit())
                .completionPrice(usage.getCompletionPrice())
                .latency(usage.getLatency())
                .build();
        }

        public Map<String, Object> toMap() {
            return Map.of(
                "total_price", totalPrice,
                "currency", currency,
                "total_tokens", totalTokens,
                "prompt_tokens", promptTokens,
                "prompt_unit_price", promptUnitPrice,
                "prompt_price_unit", promptPriceUnit,
                "prompt_price", promptPrice,
                "completion_tokens", completionTokens,
                "completion_unit_price", completionUnitPrice,
                "completion_price_unit", completionPriceUnit,
                "completion_price", completionPrice,
                "latency", latency
            );
        }
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
    public static class FunctionCallingParams {
        private String query;
        private String instruction;
        private AgentModelConfig model;
        private List<ToolEntity> tools;
        @Builder.Default
        private int maximumIterations = 3;
        private List<ContextItem> context;
    }

    @Data
    @AllArgsConstructor
    public static class ToolCall {
        private String toolCallId;
        private String toolCallName;
        private Map<String, Object> toolCallArgs;
    }
}
