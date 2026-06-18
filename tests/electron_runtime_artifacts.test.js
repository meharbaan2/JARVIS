const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
    readRuntimeArtifact,
    resolveRuntimeArtifactPath,
} = require('../AICoreClient/electron/runtimeArtifacts');

function withTempRepo(callback) {
    const root = fs.mkdtempSync(path.join(__dirname, 'tmp-artifacts-'));
    try {
        return callback(root);
    } finally {
        fs.rmSync(root, { recursive: true, force: true });
    }
}

test('reads JSON artifacts from runtime tool run storage', () => {
    withTempRepo((repoRoot) => {
        const artifactPath = path.join(repoRoot, 'AICoreClient', 'runtime_state', 'tool_runs', 'ofdmsim', 'run.json');
        fs.mkdirSync(path.dirname(artifactPath), { recursive: true });
        fs.writeFileSync(artifactPath, JSON.stringify({ source_tool: 'ofdmsim', metrics: { ber: 0 } }), 'utf8');

        const result = readRuntimeArtifact(repoRoot, 'AICoreClient/runtime_state/tool_runs/ofdmsim/run.json');

        assert.equal(result.artifact.source_tool, 'ofdmsim');
        assert.equal(result.artifact.metrics.ber, 0);
        assert.equal(result.ai_source, 'RuntimeArtifactStore');
    });
});

test('blocks artifact reads outside runtime state storage', () => {
    withTempRepo((repoRoot) => {
        const outsidePath = path.join(repoRoot, 'AICoreClient', 'PythonServer', 'ai_server.py');
        fs.mkdirSync(path.dirname(outsidePath), { recursive: true });
        fs.writeFileSync(outsidePath, 'print("nope")', 'utf8');

        assert.throws(
            () => resolveRuntimeArtifactPath(repoRoot, outsidePath),
            /outside the runtime state directory/,
        );
        assert.throws(
            () => resolveRuntimeArtifactPath(repoRoot, 'AICoreClient/runtime_state/tool_runs/../../PythonServer/ai_server.py'),
            /outside the runtime state directory/,
        );
    });
});
