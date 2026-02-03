package com.dify.agent_strategies.cot_agent.strategies;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public abstract class AgentStrategy {
    protected AgentSession session;

    public void setSession(AgentSession session) {
        this.session = session;
    }

    public abstract Iterable<AgentInvokeMessage> invoke(Map<String, Object> parameters);

    protected List<ToolEntity> initPromptTools(List<ToolEntity> tools) {
        if (tools == null) {
            return new ArrayList<>();
        }
        return new ArrayList<>(tools);
    }

    protected void updatePromptMessageTool(ToolEntity toolInstance, ToolEntity promptTool) {
        if (toolInstance == null || promptTool == null) {
            return;
        }
        promptTool.setRuntimeParameters(toolInstance.getRuntimeParameters());
        promptTool.setParameters(toolInstance.getParameters());
        promptTool.setIdentity(toolInstance.getIdentity());
        promptTool.setProviderType(toolInstance.getProviderType());
    }

    protected void recalcLlmMaxTokens(
        Object modelEntity,
        List<PromptMessage> promptMessages,
        Map<String, Object> completionParams
    ) {
        // Intentionally left blank; depends on model token calculation implementation.
    }

    protected void increaseUsage(Map<String, Optional<LLMUsage>> usageDict, LLMUsage usage) {
        if (usage == null) {
            return;
        }
        usageDict.put("usage", Optional.of(usage));
    }

    protected ToolInvokeMessage.LogMessage createLogMessage(
        String label,
        Map<String, Object> data,
        Map<String, Object> metadata,
        ToolInvokeMessage.LogMessage.LogStatus status
    ) {
        return createLogMessage(label, data, metadata, null, status);
    }

    protected ToolInvokeMessage.LogMessage createLogMessage(
        String label,
        Map<String, Object> data,
        Map<String, Object> metadata,
        ToolInvokeMessage.LogMessage parent,
        ToolInvokeMessage.LogMessage.LogStatus status
    ) {
        ToolInvokeMessage.LogMessage message = new ToolInvokeMessage.LogMessage();
        message.setLabel(label);
        message.setData(data);
        message.setMetadata(metadata);
        message.setParent(parent);
        message.setStatus(status);
        return message;
    }

    protected ToolInvokeMessage.LogMessage finishLogMessage(
        ToolInvokeMessage.LogMessage log,
        Map<String, Object> data,
        Map<String, Object> metadata
    ) {
        log.setData(data);
        log.setMetadata(metadata);
        log.setStatus(ToolInvokeMessage.LogMessage.LogStatus.FINISH);
        return log;
    }

    protected AgentInvokeMessage createTextMessage(String text) {
        ToolInvokeMessage.TextMessage message = new ToolInvokeMessage.TextMessage();
        message.setText(text);
        return message;
    }

    protected AgentInvokeMessage createJsonMessage(Map<String, Object> jsonObject) {
        ToolInvokeMessage.JsonMessage message = new ToolInvokeMessage.JsonMessage();
        message.setJsonObject(jsonObject);
        return message;
    }

    protected AgentInvokeMessage createRetrieverResourceMessage(
        List<ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource> retrieverResources,
        String context
    ) {
        ToolInvokeMessage.RetrieverResourceMessage message = new ToolInvokeMessage.RetrieverResourceMessage();
        message.setRetrieverResources(retrieverResources);
        message.setContext(context);
        return message;
    }

    protected AgentInvokeMessage createBlobMessage(byte[] blob, Map<String, Object> meta) {
        ToolInvokeMessage.BlobMessage message = new ToolInvokeMessage.BlobMessage();
        message.setBlob(blob);
        message.setMeta(meta);
        return message;
    }
}
