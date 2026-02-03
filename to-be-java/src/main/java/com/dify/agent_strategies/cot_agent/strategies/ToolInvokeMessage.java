package com.dify.agent_strategies.cot_agent.strategies;

import java.util.List;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ToolInvokeMessage implements AgentInvokeMessage {
    public enum MessageType {
        TEXT,
        LINK,
        IMAGE_LINK,
        IMAGE,
        JSON,
        BLOB
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LogMessage implements AgentInvokeMessage {
        private String label;
        private Map<String, Object> data;
        private Map<String, Object> metadata;
        private LogMessage parent;
        private LogStatus status;

        public enum LogStatus {
            START,
            FINISH
        }
    }

    @Data
    public static class TextMessage implements AgentInvokeMessage {
        private String text;
    }

    @Data
    public static class JsonMessage implements AgentInvokeMessage {
        private Map<String, Object> jsonObject;
    }

    @Data
    public static class BlobMessage implements AgentInvokeMessage {
        private byte[] blob;
        private Map<String, Object> meta;
    }

    @Data
    public static class RetrieverResourceMessage implements AgentInvokeMessage {
        private List<RetrieverResource> retrieverResources;
        private String context;

        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class RetrieverResource {
            private String content;
            private Object position;
            private Object datasetId;
            private Object datasetName;
            private Object documentId;
            private Object documentName;
            private Object dataSourceType;
            private Object segmentId;
            private Object retrieverFrom;
            private Object score;
            private Object hitCount;
            private Object wordCount;
            private Object segmentPosition;
            private Object indexNodeHash;
            private Object page;
            private Object docMetadata;
        }
    }

    private MessageType type;
    private AgentInvokeMessage message;
}
