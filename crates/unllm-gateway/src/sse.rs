#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseMessage {
    pub event_type: Option<String>,
    pub data: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseMessage> {
        self.buffer.extend_from_slice(bytes);
        let mut messages = Vec::new();
        while let Some(position) = find_separator(&self.buffer) {
            let separator_len = if self.buffer.get(position..position + 4) == Some(b"\r\n\r\n") {
                4
            } else {
                2
            };
            let frame = self.buffer.drain(..position).collect::<Vec<_>>();
            self.buffer.drain(..separator_len);
            if let Some(message) = parse_frame(&frame) {
                messages.push(message);
            }
        }
        messages
    }

    pub fn finish(&mut self) -> Option<SseMessage> {
        if self.buffer.is_empty() {
            return None;
        }
        let frame = std::mem::take(&mut self.buffer);
        parse_frame(&frame)
    }
}

fn find_separator(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .or_else(|| buffer.windows(4).position(|window| window == b"\r\n\r\n"))
}

fn parse_frame(frame: &[u8]) -> Option<SseMessage> {
    let text = String::from_utf8_lossy(frame);
    let mut event_type = None;
    let mut data = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.starts_with(':') {
            continue;
        }
        if let Some(value) = line.strip_prefix("event:") {
            event_type = Some(value.trim_start().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(value.trim_start().as_bytes());
        }
    }
    (!data.is_empty()).then_some(SseMessage { event_type, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_fragmented_utf8_and_multiline_data() {
        let raw = "event: message\ndata: {\"text\":\"你好\"}\ndata: tail\n\n".as_bytes();
        let split = raw.len() - 4;
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&raw[..split]).is_empty());
        let messages = decoder.push(&raw[split..]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].event_type.as_deref(), Some("message"));
        assert_eq!(
            String::from_utf8_lossy(&messages[0].data),
            "{\"text\":\"你好\"}\ntail"
        );
    }
}
