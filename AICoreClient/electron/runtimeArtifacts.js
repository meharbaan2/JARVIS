const fs = require('fs');
const path = require('path');

const DEFAULT_ALLOWED_SUBDIRS = [
    ['AICoreClient', 'runtime_state', 'tool_runs'],
    ['AICoreClient', 'runtime_state', 'bayesian'],
];

function resolveRuntimeArtifactPath(repoRoot, candidatePath, allowedSubdirs = DEFAULT_ALLOWED_SUBDIRS) {
    const rawPath = String(candidatePath || '').trim();
    if (!rawPath) {
        throw new Error('artifactPath is required.');
    }

    const resolvedRoot = path.resolve(repoRoot);
    const resolvedPath = path.resolve(path.isAbsolute(rawPath) ? rawPath : path.join(resolvedRoot, rawPath));
    const allowedRoots = allowedSubdirs.map((parts) => path.resolve(resolvedRoot, ...parts));
    const isAllowed = allowedRoots.some((root) => resolvedPath === root || resolvedPath.startsWith(`${root}${path.sep}`));
    if (!isAllowed) {
        throw new Error('Artifact path is outside the runtime state directory.');
    }
    return resolvedPath;
}

function readRuntimeArtifact(repoRoot, candidatePath, options = {}) {
    const artifactPath = resolveRuntimeArtifactPath(repoRoot, candidatePath, options.allowedSubdirs);
    const stats = fs.statSync(artifactPath);
    const maxBytes = options.maxBytes || 1024 * 1024;
    if (!stats.isFile()) {
        throw new Error('Artifact path is not a file.');
    }
    if (stats.size > maxBytes) {
        throw new Error('Artifact is too large to preview.');
    }

    const rawText = fs.readFileSync(artifactPath, 'utf8');
    try {
        return {
            artifact_path: artifactPath,
            artifact: JSON.parse(rawText),
            raw_text: rawText,
            ai_source: 'RuntimeArtifactStore',
        };
    } catch {
        return {
            artifact_path: artifactPath,
            artifact: null,
            raw_text: rawText,
            ai_source: 'RuntimeArtifactStore',
        };
    }
}

module.exports = {
    readRuntimeArtifact,
    resolveRuntimeArtifactPath,
};
