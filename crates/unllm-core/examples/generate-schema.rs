//! Generates or verifies the checked-in canonical JSON Schema artifacts.

use std::{fs, path::PathBuf};

use schemars::schema_for;
use unllm_core::{RequestEnvelope, ResponseEnvelope, StreamEventEnvelope};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let check = std::env::args().any(|argument| argument == "--check");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = root.join("schema/v1");
    let schemas = [
        ("request.schema.json", schema_for!(RequestEnvelope)),
        ("response.schema.json", schema_for!(ResponseEnvelope)),
        ("stream-event.schema.json", schema_for!(StreamEventEnvelope)),
    ];
    fs::create_dir_all(&output)?;
    for (name, schema) in schemas {
        let path = output.join(name);
        let serialized = format!("{}\n", serde_json::to_string_pretty(&schema)?);
        if check {
            let committed = fs::read_to_string(&path)?;
            if committed != serialized {
                return Err(format!("{} is out of date", path.display()).into());
            }
        } else {
            fs::write(path, serialized)?;
        }
    }
    Ok(())
}
