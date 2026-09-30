use std::{collections::BTreeMap, pin::Pin};

use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ContentPart, Extensions, FinishReason, ToolCall, UnifiedResponse, UnllmError, Usage};

/// Runtime-neutral boxed stream of normalized events.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, UnllmError>> + Send>>;

/// A typed incremental content payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolArguments {
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        fragment: String,
    },
    Media {
        part: ContentPart,
    },
}

/// A normalized streaming event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    ResponseStart {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        model: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        extensions: Extensions,
    },
    ContentBlockStart {
        candidate_index: u32,
        block_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        part: Option<ContentPart>,
    },
    ContentDelta {
        candidate_index: u32,
        block_index: u32,
        delta: ContentDelta,
    },
    ContentBlockStop {
        candidate_index: u32,
        block_index: u32,
    },
    Usage {
        usage: Usage,
    },
    Finish {
        candidate_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<FinishReason>,
    },
    Error {
        error: UnllmError,
    },
    Raw {
        provider: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        event_type: Option<String>,
        payload: Value,
    },
}

/// Versioned serialized stream event contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StreamEventEnvelope {
    /// Canonical schema version. Version 1 is currently supported.
    pub schema_version: u32,
    /// Canonical stream event.
    pub value: StreamEvent,
}

/// Reconstructs a useful complete response from normalized events.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    id: Option<String>,
    model: Option<String>,
    text: BTreeMap<(u32, u32), String>,
    reasoning: BTreeMap<(u32, u32), String>,
    tool_calls: BTreeMap<(u32, u32), (String, Option<String>, String)>,
    finish: BTreeMap<u32, Option<FinishReason>>,
    usage: Option<Usage>,
    extensions: Extensions,
}

impl StreamAccumulator {
    /// Pushes one event into the accumulator.
    pub fn push(&mut self, event: StreamEvent) -> Result<(), UnllmError> {
        match event {
            StreamEvent::ResponseStart {
                id,
                model,
                extensions,
            } => {
                self.id = id;
                self.model = Some(model);
                self.extensions.extend(extensions);
            }
            StreamEvent::ContentDelta {
                candidate_index,
                block_index,
                delta,
            } => match delta {
                ContentDelta::Text { text } => self
                    .text
                    .entry((candidate_index, block_index))
                    .or_default()
                    .push_str(&text),
                ContentDelta::Reasoning { text } => self
                    .reasoning
                    .entry((candidate_index, block_index))
                    .or_default()
                    .push_str(&text),
                ContentDelta::ToolArguments {
                    call_id,
                    name,
                    fragment,
                } => {
                    let entry = self
                        .tool_calls
                        .entry((candidate_index, block_index))
                        .or_insert((call_id, name.clone(), String::new()));
                    if entry.1.is_none() {
                        entry.1 = name;
                    }
                    entry.2.push_str(&fragment);
                }
                ContentDelta::Media { .. } => {}
            },
            StreamEvent::Usage { usage } => self.usage = Some(usage),
            StreamEvent::Finish {
                candidate_index,
                reason,
            } => {
                self.finish.insert(candidate_index, reason);
            }
            StreamEvent::Error { error } => return Err(error),
            StreamEvent::ContentBlockStart { .. }
            | StreamEvent::ContentBlockStop { .. }
            | StreamEvent::Raw { .. } => {}
        }
        Ok(())
    }

    /// Finishes accumulation and validates completed tool JSON.
    pub fn finish(self) -> Result<UnifiedResponse, UnllmError> {
        let model = self.model.ok_or_else(|| {
            UnllmError::invalid(
                "missing_stream_start",
                "Stream did not include a response start",
            )
        })?;
        let max_candidate = self
            .text
            .keys()
            .chain(self.reasoning.keys())
            .chain(self.tool_calls.keys())
            .map(|(candidate, _)| *candidate)
            .chain(self.finish.keys().copied())
            .max()
            .unwrap_or(0);
        let mut candidates = Vec::new();
        for candidate_index in 0..=max_candidate {
            let mut blocks: Vec<(u32, ContentPart)> = Vec::new();
            for ((candidate, block), text) in &self.text {
                if *candidate == candidate_index {
                    blocks.push((*block, ContentPart::text(text.clone())));
                }
            }
            for ((candidate, block), text) in &self.reasoning {
                if *candidate == candidate_index {
                    blocks.push((
                        *block,
                        ContentPart::Reasoning {
                            text: Some(text.clone()),
                            extensions: Extensions::new(),
                        },
                    ));
                }
            }
            for ((candidate, block), (id, name, arguments)) in &self.tool_calls {
                if *candidate == candidate_index {
                    let arguments = serde_json::from_str(arguments).map_err(|error| {
                        UnllmError::invalid(
                            "invalid_tool_arguments",
                            format!("Completed tool arguments are not valid JSON: {error}"),
                        )
                    })?;
                    blocks.push((
                        *block,
                        ContentPart::ToolCall {
                            call: ToolCall {
                                id: id.clone(),
                                name: name.clone().unwrap_or_default(),
                                arguments,
                                extensions: Extensions::new(),
                            },
                        },
                    ));
                }
            }
            blocks.sort_by_key(|(index, _)| *index);
            candidates.push(crate::Candidate {
                index: candidate_index,
                id: None,
                content: blocks.into_iter().map(|(_, part)| part).collect(),
                finish_reason: self.finish.get(&candidate_index).cloned().flatten(),
                extensions: Extensions::new(),
            });
        }
        Ok(UnifiedResponse::Generate(crate::GenerateResponse {
            id: self.id,
            model,
            candidates,
            usage: self.usage,
            extensions: self.extensions,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_fragmented_tool_arguments() {
        let mut accumulator = StreamAccumulator::default();
        accumulator
            .push(StreamEvent::ResponseStart {
                id: Some("r1".into()),
                model: "model".into(),
                extensions: Extensions::new(),
            })
            .unwrap();
        for fragment in ["{\"city\":", "\"Paris\"}"] {
            accumulator
                .push(StreamEvent::ContentDelta {
                    candidate_index: 0,
                    block_index: 0,
                    delta: ContentDelta::ToolArguments {
                        call_id: "call-0".into(),
                        name: Some("weather".into()),
                        fragment: fragment.into(),
                    },
                })
                .unwrap();
        }
        let UnifiedResponse::Generate(response) = accumulator.finish().unwrap() else {
            panic!("expected a generation response");
        };
        let ContentPart::ToolCall { call } = &response.candidates[0].content[0] else {
            panic!("expected a tool call");
        };
        assert_eq!(call.arguments["city"], "Paris");
    }
}
