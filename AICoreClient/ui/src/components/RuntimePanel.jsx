import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    Activity,
    AlertTriangle,
    CheckCircle2,
    Database,
    History,
    Play,
    RefreshCw,
    RotateCcw,
    Shield,
    Terminal,
    Wrench,
    X,
} from 'lucide-react';

const panelStyle = {
    position: 'fixed',
    top: '96px',
    right: '2rem',
    bottom: '96px',
    width: 'min(440px, calc(100vw - 4rem))',
    zIndex: 1200,
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid rgba(0, 212, 255, 0.24)',
    borderRadius: '8px',
    background: 'rgba(7, 12, 28, 0.94)',
    backdropFilter: 'blur(18px)',
    boxShadow: '0 18px 48px rgba(0, 0, 0, 0.42)',
    overflow: 'hidden',
};

const headerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    padding: '14px 16px',
    borderBottom: '1px solid rgba(0, 212, 255, 0.16)',
};

const iconButtonStyle = {
    width: '34px',
    height: '34px',
    borderRadius: '8px',
    border: '1px solid rgba(0, 212, 255, 0.22)',
    background: 'rgba(13, 24, 48, 0.8)',
    color: '#dff9ff',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
};

const sectionStyle = {
    padding: '14px 16px',
    borderBottom: '1px solid rgba(0, 212, 255, 0.1)',
};

const toolRowStyle = {
    border: '1px solid rgba(0, 212, 255, 0.14)',
    borderRadius: '8px',
    padding: '10px',
    background: 'rgba(13, 24, 48, 0.56)',
};

const mutedTextStyle = {
    color: 'rgba(224, 242, 254, 0.66)',
    fontSize: '0.78rem',
    lineHeight: 1.45,
};

const inputStyle = {
    width: '100%',
    border: '1px solid rgba(0, 212, 255, 0.18)',
    borderRadius: '8px',
    background: 'rgba(5, 12, 28, 0.84)',
    color: '#e0f2fe',
    padding: '8px 9px',
    fontSize: '0.78rem',
    outline: 'none',
};

const runButtonStyle = {
    border: '1px solid rgba(34, 211, 238, 0.36)',
    borderRadius: '8px',
    padding: '8px 11px',
    background: 'linear-gradient(135deg, rgba(34, 211, 238, 0.9), rgba(59, 130, 246, 0.9))',
    color: '#ffffff',
    cursor: 'pointer',
    fontSize: '0.78rem',
    fontWeight: 700,
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
};

const safeCommands = [
    'runtime health',
    'list tools',
    'list models',
    'refresh project memory',
    'recent tool runs',
    'inspect FolderGuardian project',
    'run OFDMSim self test',
    'run OFDMSim AWGN QPSK MMSE at 18 dB for 10 frames seed 42 and update Bayesian confidence',
    'compare OFDMSim EMI AWGN QPSK at 18 dB for 100 frames EMI probability 0.003 amplitude 15',
    'assess OFDMSim confidence',
    'recent execution plans',
];

function statusColor(available) {
    return available ? '#22c55e' : '#f59e0b';
}

function parseMetrics(metricsJson) {
    if (!metricsJson) return null;
    try {
        return JSON.parse(metricsJson);
    } catch {
        return null;
    }
}

function riskColor(level) {
    const text = String(level || '').toLowerCase();
    if (text === 'critical' || text === 'high') return '#f87171';
    if (text === 'elevated' || text === 'medium') return '#fbbf24';
    if (text === 'low') return '#22c55e';
    return '#67e8f9';
}

function compactRunSummary(run, metrics) {
    if (!metrics) return '';
    if (run.tool_name === 'security_suite') {
        if (metrics.risk_level || metrics.risk_score !== undefined) {
            return `${metrics.risk_level || 'unknown'} risk, score ${metrics.risk_score ?? 'unknown'}/100`;
        }
        if (metrics.returncode !== undefined) {
            return `Defender return code ${metrics.returncode}`;
        }
    }
    if (metrics.ber !== undefined) return `BER ${metrics.ber}`;
    if (metrics.baseline_ber !== undefined) return `Baseline BER ${metrics.baseline_ber}`;
    if (metrics.manifest_count !== undefined) return `${metrics.manifest_count} manifests audited`;
    if (Array.isArray(metrics.detected_features)) return metrics.detected_features.slice(0, 3).join(', ');
    return '';
}

function isEnumSpec(spec) {
    const parts = String(spec || '').split('|').map((part) => part.trim()).filter(Boolean);
    const primitive = new Set(['path', 'json', 'string', 'integer', 'number', 'boolean', 'list']);
    return parts.length > 1 && parts.some((part) => !primitive.has(part.toLowerCase()));
}

function schemaKind(spec) {
    const text = String(spec || '').toLowerCase();
    if (isEnumSpec(spec)) return 'enum';
    if (text.includes('boolean')) return 'boolean';
    if (text.includes('integer')) return 'integer';
    if (text.includes('number')) return 'number';
    if (text.includes('json') || text.includes('list')) return 'json';
    return 'text';
}

function normalizeParamValue(rawValue, spec) {
    const kind = schemaKind(spec);
    if (kind === 'boolean') {
        return rawValue === true ? true : undefined;
    }
    if (rawValue === undefined || rawValue === null || rawValue === '') {
        return undefined;
    }
    if (kind === 'integer') {
        const parsed = Number.parseInt(rawValue, 10);
        return Number.isNaN(parsed) ? rawValue : parsed;
    }
    if (kind === 'number') {
        const parsed = Number.parseFloat(rawValue);
        return Number.isNaN(parsed) ? rawValue : parsed;
    }
    if (kind === 'json') {
        try {
            return JSON.parse(rawValue);
        } catch {
            return rawValue;
        }
    }
    return rawValue;
}

export default function RuntimePanel({ open, onClose }) {
    const [snapshot, setSnapshot] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [commandOutput, setCommandOutput] = useState('');
    const [commandLoading, setCommandLoading] = useState(false);
    const [runnerToolName, setRunnerToolName] = useState('');
    const [runnerCapabilityName, setRunnerCapabilityName] = useState('');
    const [runnerParams, setRunnerParams] = useState({});
    const [memoryQuery, setMemoryQuery] = useState('OFDM BER quiet run');
    const [memoryProject, setMemoryProject] = useState('');
    const [memoryLimit, setMemoryLimit] = useState(5);
    const [memoryIndexLimit, setMemoryIndexLimit] = useState(48);
    const [memoryHits, setMemoryHits] = useState([]);
    const [memorySearchLoading, setMemorySearchLoading] = useState(false);
    const [memoryIndexLoading, setMemoryIndexLoading] = useState(false);
    const [selectedArtifact, setSelectedArtifact] = useState(null);
    const [artifactLoading, setArtifactLoading] = useState(false);
    const [selectedPlan, setSelectedPlan] = useState(null);
    const [modelProbe, setModelProbe] = useState(null);
    const [modelProbeLoading, setModelProbeLoading] = useState(false);

    const refresh = useCallback(async () => {
        if (!window.electronAPI?.getRuntimeSnapshot) {
            setError('Runtime bridge is not available in this window.');
            return;
        }

        setLoading(true);
        setError('');
        try {
            const nextSnapshot = await window.electronAPI.getRuntimeSnapshot();
            setSnapshot(nextSnapshot);
            if (nextSnapshot.error) {
                setError(nextSnapshot.error);
            }
        } catch (err) {
            setError(err.message || 'Runtime snapshot failed.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (open) {
            refresh();
        }
    }, [open, refresh]);

    const tools = useMemo(() => snapshot?.tools || [], [snapshot]);
    const modelSnapshot = snapshot?.models || {};
    const models = modelSnapshot.models || [];
    const recentRuns = snapshot?.recent_tool_runs || [];
    const recentPlans = snapshot?.recent_execution_plans || [];
    const memory = snapshot?.memory || {};
    const memoryProjectCounts = memory.project_document_counts || {};
    const availableCount = snapshot?.available_tool_count || 0;
    const toolCount = snapshot?.tool_count || tools.length;
    const selectedTool = tools.find((tool) => tool.name === runnerToolName) || tools[0] || null;
    const selectedCapability = selectedTool?.capabilities?.find((capability) => capability.name === runnerCapabilityName)
        || selectedTool?.capabilities?.[0]
        || null;
    const selectedSchema = selectedCapability?.input_schema || {};
    const memoryProjectOptions = useMemo(() => {
        const options = [{ label: 'All memory', value: '' }];
        tools.forEach((tool) => {
            options.push({ label: `${tool.name} project docs`, value: tool.name });
            options.push({ label: `${tool.name} tool runs`, value: `tool:${tool.name}` });
        });
        Object.keys(memoryProjectCounts).forEach((project) => {
            if (!options.some((option) => option.value === project)) {
                options.push({ label: project, value: project });
            }
        });
        return options;
    }, [tools, memoryProjectCounts]);

    const capabilityCount = tools.reduce((total, tool) => total + (tool.capabilities?.length || 0), 0);

    useEffect(() => {
        if (!tools.length) return;
        if (!runnerToolName || !tools.some((tool) => tool.name === runnerToolName)) {
            const firstTool = tools[0];
            setRunnerToolName(firstTool.name);
            setRunnerCapabilityName(firstTool.capabilities?.[0]?.name || '');
            setRunnerParams({});
            return;
        }
        if (selectedTool && (!runnerCapabilityName || !selectedTool.capabilities?.some((capability) => capability.name === runnerCapabilityName))) {
            setRunnerCapabilityName(selectedTool.capabilities?.[0]?.name || '');
            setRunnerParams({});
        }
    }, [tools, runnerToolName, runnerCapabilityName, selectedTool]);

    const runCommand = async (command) => {
        if (!window.electronAPI?.sendRuntimeCommand) {
            setCommandOutput('Runtime command bridge is not available.');
            return;
        }

        setCommandLoading(true);
        try {
            const result = await window.electronAPI.sendRuntimeCommand(command);
            setCommandOutput(result.response_text || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'Runtime command failed.');
        } finally {
            setCommandLoading(false);
        }
    };

    const updateRunnerParam = (name, value) => {
        setRunnerParams((current) => ({
            ...current,
            [name]: value,
        }));
    };

    const runSelectedCapability = async () => {
        if (!selectedTool || !selectedCapability) return;
        const params = {};
        Object.entries(selectedSchema).forEach(([name, spec]) => {
            const value = normalizeParamValue(runnerParams[name], spec);
            if (value !== undefined) {
                params[name] = value;
            }
        });

        if (!window.electronAPI?.runRuntimeCapability) {
            await runCommand(`run capability ${selectedTool.name}.${selectedCapability.name} ${JSON.stringify(params)}`);
            return;
        }

        setCommandLoading(true);
        try {
            const result = await window.electronAPI.runRuntimeCapability({
                toolName: selectedTool.name,
                capabilityName: selectedCapability.name,
                params,
            });
            setCommandOutput(result.response_text || result.error_message || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'Runtime capability failed.');
        } finally {
            setCommandLoading(false);
        }
    };

    const searchMemory = async () => {
        if (!window.electronAPI?.searchRuntimeMemory) {
            setCommandOutput('Direct memory search bridge is not available.');
            return;
        }
        if (!memoryQuery.trim()) {
            setCommandOutput('Enter a memory search query first.');
            return;
        }

        setMemorySearchLoading(true);
        try {
            const result = await window.electronAPI.searchRuntimeMemory({
                query: memoryQuery.trim(),
                project: memoryProject.trim(),
                limit: Number(memoryLimit) || 5,
            });
            setMemoryHits(result.hits || []);
            if (result.response_text && !(result.hits || []).length) {
                setCommandOutput(result.response_text);
            }
        } catch (err) {
            setCommandOutput(err.message || 'Memory search failed.');
        } finally {
            setMemorySearchLoading(false);
        }
    };

    const indexMemory = async () => {
        if (!window.electronAPI?.indexRuntimeMemory) {
            await runCommand('refresh project memory');
            return;
        }

        setMemoryIndexLoading(true);
        try {
            const result = await window.electronAPI.indexRuntimeMemory({
                projectNames: memoryProject.trim() ? [memoryProject.trim()] : [],
                maxFilesPerProject: Number(memoryIndexLimit) || 48,
            });
            if (result.response_text) {
                setCommandOutput(result.response_text);
            } else {
                const lines = [`Indexed ${result.total_indexed_document_count || 0} documents.`];
                (result.results || []).forEach((item) => {
                    lines.push(`- ${item.project}: ${item.indexed_document_count} (${item.status})`);
                });
                setCommandOutput(lines.join('\n'));
            }
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'Project memory indexing failed.');
        } finally {
            setMemoryIndexLoading(false);
        }
    };

    const loadModel = async (modelName) => {
        if (!window.electronAPI?.loadRuntimeModel) {
            await runCommand(`load ${modelName}`);
            return;
        }
        setCommandLoading(true);
        try {
            const result = await window.electronAPI.loadRuntimeModel({ modelName });
            setCommandOutput(result.message || `${modelName} load request completed.`);
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || `${modelName} load request failed.`);
        } finally {
            setCommandLoading(false);
        }
    };

    const unloadModel = async (modelName) => {
        if (!window.electronAPI?.unloadRuntimeModel) {
            await runCommand(`unload ${modelName}`);
            return;
        }
        setCommandLoading(true);
        try {
            const result = await window.electronAPI.unloadRuntimeModel({ modelName });
            setCommandOutput(result.message || `${modelName} unload request completed.`);
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || `${modelName} unload request failed.`);
        } finally {
            setCommandLoading(false);
        }
    };

    const viewArtifact = async (run) => {
        if (!run?.artifact_path) return;
        if (!window.electronAPI?.readRuntimeArtifact) {
            setCommandOutput('Runtime artifact bridge is not available.');
            return;
        }

        setArtifactLoading(true);
        try {
            const result = await window.electronAPI.readRuntimeArtifact({ artifactPath: run.artifact_path });
            if (result.error_message) {
                setCommandOutput(result.error_message);
                setSelectedArtifact(null);
            } else {
                setSelectedArtifact(result);
            }
        } catch (err) {
            setCommandOutput(err.message || 'Runtime artifact read failed.');
            setSelectedArtifact(null);
        } finally {
            setArtifactLoading(false);
        }
    };

    const assessOfdmsimConfidence = async () => {
        if (!window.electronAPI?.runRuntimeCapability) {
            await runCommand('assess OFDMSim confidence');
            return;
        }

        setCommandLoading(true);
        try {
            const result = await window.electronAPI.runRuntimeCapability({
                toolName: 'bayesian_engine',
                capabilityName: 'assess_tool_confidence',
                params: { tool_name: 'ofdmsim' },
            });
            setCommandOutput(result.response_text || result.error_message || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'OFDMSim confidence assessment failed.');
        } finally {
            setCommandLoading(false);
        }
    };

    const showBayesianBreakdown = async (toolName = 'ofdmsim') => {
        if (!window.electronAPI?.runRuntimeCapability) {
            await runCommand(`bayesian ${toolName} evidence breakdown`);
            return;
        }

        setCommandLoading(true);
        try {
            const result = await window.electronAPI.runRuntimeCapability({
                toolName: 'bayesian_engine',
                capabilityName: 'tool_confidence_breakdown',
                params: { tool_name: toolName },
            });
            setCommandOutput(result.response_text || result.error_message || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || `${toolName} Bayesian breakdown failed.`);
        } finally {
            setCommandLoading(false);
        }
    };

    const prepareQuarantineFromArtifact = async () => {
        const artifact = selectedArtifact?.artifact || {};
        const metrics = artifact.metrics || {};
        const targetPath = artifact.target_path || metrics.target_path;
        if (!targetPath) {
            setCommandOutput('Selected artifact does not include a target path to quarantine.');
            return;
        }
        setCommandLoading(true);
        try {
            const result = await window.electronAPI.runRuntimeCapability({
                toolName: 'security_suite',
                capabilityName: 'quarantine_path',
                params: { target_path: targetPath },
            });
            setCommandOutput(result.response_text || result.error_message || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'Quarantine preparation failed.');
        } finally {
            setCommandLoading(false);
        }
    };

    const prepareRestoreFromArtifact = async () => {
        const artifact = selectedArtifact?.artifact || {};
        const metrics = artifact.metrics || {};
        const metadataPath = artifact.metadata_path || metrics.metadata_path;
        if (!metadataPath) {
            setCommandOutput('Selected artifact does not include quarantine metadata to restore.');
            return;
        }
        setCommandLoading(true);
        try {
            const result = await window.electronAPI.runRuntimeCapability({
                toolName: 'security_suite',
                capabilityName: 'restore_quarantine',
                params: { metadata_path: metadataPath },
            });
            setCommandOutput(result.response_text || result.error_message || 'No response.');
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || 'Restore preparation failed.');
        } finally {
            setCommandLoading(false);
        }
    };

    const probeModel = async (modelName) => {
        if (!window.electronAPI?.probeRuntimeModel) {
            setCommandOutput('Runtime model probe bridge is not available.');
            return;
        }

        setModelProbeLoading(true);
        try {
            const result = await window.electronAPI.probeRuntimeModel({ modelName, prompts: [] });
            setModelProbe(result);
            setCommandOutput(result.message || `${modelName} probe completed.`);
            await refresh();
        } catch (err) {
            setCommandOutput(err.message || `${modelName} probe failed.`);
            setModelProbe(null);
        } finally {
            setModelProbeLoading(false);
        }
    };

    const renderRunnerField = (name, spec) => {
        const kind = schemaKind(spec);
        const value = runnerParams[name] ?? '';
        if (kind === 'enum') {
            const options = String(spec).split('|').map((part) => part.trim()).filter(Boolean);
            return (
                <select value={value} onChange={(event) => updateRunnerParam(name, event.target.value)} style={inputStyle}>
                    <option value="">Default</option>
                    {options.map((option) => (
                        <option key={option} value={option}>{option}</option>
                    ))}
                </select>
            );
        }
        if (kind === 'boolean') {
            return (
                <label style={{ ...mutedTextStyle, display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                    <input
                        type="checkbox"
                        checked={Boolean(runnerParams[name])}
                        onChange={(event) => updateRunnerParam(name, event.target.checked)}
                    />
                    enabled
                </label>
            );
        }
        if (kind === 'json') {
            return (
                <textarea
                    value={value}
                    onChange={(event) => updateRunnerParam(name, event.target.value)}
                    rows={3}
                    style={{ ...inputStyle, resize: 'vertical' }}
                />
            );
        }
        return (
            <input
                type={kind === 'number' || kind === 'integer' ? 'number' : 'text'}
                value={value}
                onChange={(event) => updateRunnerParam(name, event.target.value)}
                style={inputStyle}
            />
        );
    };

    if (!open) return null;

    return (
        <aside style={panelStyle} aria-label="Runtime tools panel">
            <div style={headerStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Activity style={{ width: '18px', height: '18px', color: '#22d3ee' }} />
                    <div>
                        <div style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.95rem' }}>
                            Hivemind Runtime
                        </div>
                        <div style={mutedTextStyle}>
                            {loading ? 'Refreshing...' : `${availableCount}/${toolCount} tools available`}
                        </div>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        type="button"
                        onClick={refresh}
                        title="Refresh runtime snapshot"
                        style={iconButtonStyle}
                        disabled={loading}
                    >
                        <RefreshCw style={{ width: '16px', height: '16px' }} />
                    </button>
                    <button type="button" onClick={onClose} title="Close runtime panel" style={iconButtonStyle}>
                        <X style={{ width: '16px', height: '16px' }} />
                    </button>
                </div>
            </div>

            <div style={{ overflowY: 'auto', flex: 1 }}>
                {error && (
                    <div style={{ ...sectionStyle, color: '#fecaca', display: 'flex', gap: '10px' }}>
                        <AlertTriangle style={{ width: '18px', height: '18px', flexShrink: 0 }} />
                        <span style={{ fontSize: '0.82rem', lineHeight: 1.45 }}>{error}</span>
                    </div>
                )}

                <div style={sectionStyle}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                        <div>
                            <div style={{ color: '#67e8f9', fontWeight: 700 }}>{toolCount}</div>
                            <div style={mutedTextStyle}>Tools</div>
                        </div>
                        <div>
                            <div style={{ color: '#67e8f9', fontWeight: 700 }}>{capabilityCount}</div>
                            <div style={mutedTextStyle}>Capabilities</div>
                        </div>
                        <div>
                            <div style={{ color: '#67e8f9', fontWeight: 700 }}>
                                {memory.project_document_chunk_count || memory.project_document_count || 0}
                            </div>
                            <div style={mutedTextStyle}>Chunks</div>
                        </div>
                    </div>
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Play style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Capability Runner</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                        <select
                            value={selectedTool?.name || ''}
                            onChange={(event) => {
                                const nextTool = tools.find((tool) => tool.name === event.target.value);
                                setRunnerToolName(event.target.value);
                                setRunnerCapabilityName(nextTool?.capabilities?.[0]?.name || '');
                                setRunnerParams({});
                            }}
                            style={inputStyle}
                        >
                            {tools.map((tool) => (
                                <option key={tool.name} value={tool.name}>{tool.name}</option>
                            ))}
                        </select>
                        <select
                            value={selectedCapability?.name || ''}
                            onChange={(event) => {
                                setRunnerCapabilityName(event.target.value);
                                setRunnerParams({});
                            }}
                            style={inputStyle}
                        >
                            {(selectedTool?.capabilities || []).map((capability) => (
                                <option key={capability.name} value={capability.name}>{capability.name}</option>
                            ))}
                        </select>
                    </div>
                    {selectedCapability && (
                        <div style={{ ...mutedTextStyle, marginTop: '8px' }}>{selectedCapability.description}</div>
                    )}
                    <div style={{ display: 'grid', gap: '8px', marginTop: '10px' }}>
                        {Object.entries(selectedSchema).map(([name, spec]) => (
                            <label key={name} style={{ display: 'grid', gap: '5px' }}>
                                <span style={mutedTextStyle}>{name} ({spec})</span>
                                {renderRunnerField(name, spec)}
                            </label>
                        ))}
                        {selectedCapability && Object.keys(selectedSchema).length === 0 && (
                            <div style={mutedTextStyle}>No parameters.</div>
                        )}
                    </div>
                    <button
                        type="button"
                        onClick={runSelectedCapability}
                        disabled={!selectedTool || !selectedCapability || commandLoading}
                        style={{ ...runButtonStyle, marginTop: '10px', opacity: commandLoading ? 0.65 : 1 }}
                    >
                        <Play style={{ width: '14px', height: '14px' }} />
                        Run
                    </button>
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Activity style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Models</span>
                    </div>
                    <div style={{ display: 'grid', gap: '8px' }}>
                        {models.map((model) => (
                            <div key={model.name} style={toolRowStyle}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
                                    <div style={{ minWidth: 0 }}>
                                        <div style={{ color: '#e0f2fe', fontWeight: 700, fontSize: '0.84rem' }}>
                                            {model.name}
                                        </div>
                                        <div style={mutedTextStyle}>{model.kind} - {model.provider}</div>
                                    </div>
                                    <div
                                        style={{
                                            color: statusColor(model.available),
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '4px',
                                            fontSize: '0.75rem',
                                            flexShrink: 0,
                                        }}
                                    >
                                        {model.available ? (
                                            <CheckCircle2 style={{ width: '14px', height: '14px' }} />
                                        ) : (
                                            <AlertTriangle style={{ width: '14px', height: '14px' }} />
                                        )}
                                        {model.available ? 'available' : model.status}
                                    </div>
                                </div>
                                <div style={{ ...mutedTextStyle, marginTop: '8px' }}>{model.purpose}</div>
                                <div style={{ ...mutedTextStyle, marginTop: '8px', wordBreak: 'break-all' }}>
                                    {model.model_name}
                                </div>
                                {model.name === 'phi3' && (
                                    <div style={{ display: 'flex', gap: '8px', marginTop: '10px', flexWrap: 'wrap' }}>
                                        <button
                                            type="button"
                                            onClick={() => loadModel(model.name)}
                                            disabled={commandLoading || model.available}
                                            style={{ ...runButtonStyle, opacity: commandLoading || model.available ? 0.55 : 1 }}
                                        >
                                            Load
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => unloadModel(model.name)}
                                            disabled={commandLoading || (!model.available && model.status === 'unloaded')}
                                            style={{ ...runButtonStyle, opacity: commandLoading || (!model.available && model.status === 'unloaded') ? 0.55 : 1 }}
                                        >
                                            Unload
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => probeModel(model.name)}
                                            disabled={modelProbeLoading}
                                            style={{ ...runButtonStyle, opacity: modelProbeLoading ? 0.55 : 1 }}
                                        >
                                            Probe
                                        </button>
                                    </div>
                                )}
                            </div>
                        ))}
                        {models.length === 0 && (
                            <div style={mutedTextStyle}>Model registry snapshot is not available yet.</div>
                        )}
                    </div>
                    {modelProbe && (
                        <div style={{ ...toolRowStyle, marginTop: '10px' }}>
                            <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                Model Probe: {modelProbe.model?.name || 'unknown'}
                            </div>
                            <div style={mutedTextStyle}>{modelProbe.status} - {modelProbe.message}</div>
                            <div style={{ display: 'grid', gap: '8px', marginTop: '8px' }}>
                                {(modelProbe.results || []).map((item, index) => (
                                    <div key={`${item.category}-${index}`} style={{ borderTop: '1px solid rgba(0, 212, 255, 0.1)', paddingTop: '8px' }}>
                                        <div style={{ color: '#e0f2fe', fontSize: '0.78rem', fontWeight: 700 }}>
                                            {item.category} - {item.status} - {Math.round(item.latency_ms || 0)} ms
                                        </div>
                                        <div style={{ ...mutedTextStyle, marginTop: '4px' }}>Prompt: {item.prompt}</div>
                                        {item.expected_hint && (
                                            <div style={mutedTextStyle}>Expected hint: {item.expected_hint}</div>
                                        )}
                                        {item.warning && (
                                            <div style={{ ...mutedTextStyle, color: '#fde68a' }}>Warning: {item.warning}</div>
                                        )}
                                        <div style={{ ...mutedTextStyle, marginTop: '4px', color: '#dff9ff' }}>
                                            {item.response_text}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Wrench style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Tools</span>
                    </div>
                    <div style={{ display: 'grid', gap: '8px' }}>
                        {tools.map((tool) => (
                            <div key={tool.name} style={toolRowStyle}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
                                    <div style={{ minWidth: 0 }}>
                                        <div style={{ color: '#e0f2fe', fontWeight: 700, fontSize: '0.84rem' }}>
                                            {tool.name}
                                        </div>
                                        <div style={mutedTextStyle}>{tool.runtime}</div>
                                    </div>
                                    <div
                                        style={{
                                            color: statusColor(tool.available),
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '4px',
                                            fontSize: '0.75rem',
                                            flexShrink: 0,
                                        }}
                                    >
                                        {tool.available ? (
                                            <CheckCircle2 style={{ width: '14px', height: '14px' }} />
                                        ) : (
                                            <AlertTriangle style={{ width: '14px', height: '14px' }} />
                                        )}
                                        {tool.available ? 'available' : 'missing'}
                                    </div>
                                </div>
                                <div style={{ ...mutedTextStyle, marginTop: '8px' }}>{tool.description}</div>
                                <div style={{ ...mutedTextStyle, marginTop: '8px' }}>
                                    {(tool.capabilities || []).map((capability) => capability.name).join(', ')}
                                </div>
                            </div>
                        ))}
                        {!loading && tools.length === 0 && (
                            <div style={mutedTextStyle}>No runtime tools are registered yet.</div>
                        )}
                    </div>
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Database style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Memory</span>
                    </div>
                    <div style={mutedTextStyle}>
                        {memory.conversation_count || 0} messages, {memory.summary_count || 0} summaries,{' '}
                        {memory.tool_run_count || 0} tool runs, {memory.execution_plan_count || 0} plans
                    </div>
                    <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                        {memory.project_document_count || 0} docs, {memory.project_document_chunk_count || 0} retrieval chunks,{' '}
                        {memory.embedding_dimensions || 0} dimensions
                    </div>
                    <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                        Embeddings: {memory.embedding_provider || 'unknown'}
                    </div>
                    <div style={{ ...mutedTextStyle, marginTop: '6px', wordBreak: 'break-all' }}>
                        {memory.database_path || 'Memory database path unavailable.'}
                    </div>
                    {Object.keys(memoryProjectCounts).length > 0 && (
                        <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                            Projects:{' '}
                            {Object.entries(memoryProjectCounts)
                                .slice(0, 5)
                                .map(([project, count]) => `${project} ${count}`)
                                .join(', ')}
                        </div>
                    )}
                    <div style={{ display: 'grid', gap: '8px', marginTop: '12px' }}>
                        <input
                            value={memoryQuery}
                            onChange={(event) => setMemoryQuery(event.target.value)}
                            placeholder="Search project/tool memory"
                            style={inputStyle}
                        />
                        <select
                            value={memoryProject}
                            onChange={(event) => setMemoryProject(event.target.value)}
                            style={inputStyle}
                        >
                            {memoryProjectOptions.map((option) => (
                                <option key={option.value || 'all'} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            <label style={mutedTextStyle}>
                                Search hits
                                <input
                                    type="number"
                                    min="1"
                                    max="20"
                                    value={memoryLimit}
                                    onChange={(event) => setMemoryLimit(event.target.value)}
                                    style={{ ...inputStyle, marginTop: '4px' }}
                                />
                            </label>
                            <label style={mutedTextStyle}>
                                Index files/tool
                                <input
                                    type="number"
                                    min="4"
                                    max="200"
                                    value={memoryIndexLimit}
                                    onChange={(event) => setMemoryIndexLimit(event.target.value)}
                                    style={{ ...inputStyle, marginTop: '4px' }}
                                />
                            </label>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            <button
                                type="button"
                                onClick={searchMemory}
                                disabled={memorySearchLoading}
                                style={{ ...runButtonStyle, opacity: memorySearchLoading ? 0.65 : 1 }}
                            >
                                Search Memory
                            </button>
                            <button
                                type="button"
                                onClick={indexMemory}
                                disabled={memoryIndexLoading}
                                style={{ ...runButtonStyle, opacity: memoryIndexLoading ? 0.65 : 1 }}
                            >
                                Index Memory
                            </button>
                        </div>
                        {memoryHits.length > 0 && (
                            <div style={{ display: 'grid', gap: '8px' }}>
                                {memoryHits.map((hit, index) => (
                                    <div key={`${hit.project}-${hit.source_path}-${hit.chunk_index}-${index}`} style={toolRowStyle}>
                                        <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                            {hit.project} / {hit.title || hit.source_path}
                                        </div>
                                        <div style={{ ...mutedTextStyle, wordBreak: 'break-all' }}>
                                            {hit.source_path} chunk {hit.chunk_index} - score {Number(hit.score || 0).toFixed(3)}
                                        </div>
                                        <div style={{ ...mutedTextStyle, marginTop: '6px' }}>{hit.snippet}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginBottom: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <History style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                            <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Recent Runs</span>
                        </div>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                            <button
                                type="button"
                                onClick={assessOfdmsimConfidence}
                                disabled={commandLoading}
                                style={{
                                    border: '1px solid rgba(34, 211, 238, 0.28)',
                                    borderRadius: '8px',
                                    padding: '6px 8px',
                                    background: 'rgba(13, 24, 48, 0.8)',
                                    color: '#dff9ff',
                                    cursor: 'pointer',
                                    fontSize: '0.72rem',
                                }}
                            >
                                Assess OFDMSim
                            </button>
                            <button
                                type="button"
                                onClick={() => showBayesianBreakdown('ofdmsim')}
                                disabled={commandLoading}
                                style={{
                                    border: '1px solid rgba(34, 211, 238, 0.28)',
                                    borderRadius: '8px',
                                    padding: '6px 8px',
                                    background: 'rgba(13, 24, 48, 0.8)',
                                    color: '#dff9ff',
                                    cursor: 'pointer',
                                    fontSize: '0.72rem',
                                }}
                            >
                                Breakdown
                            </button>
                        </div>
                    </div>
                    <div style={{ display: 'grid', gap: '8px' }}>
                        {recentRuns.map((run, index) => {
                            const metrics = parseMetrics(run.metrics_json);
                            return (
                                <div key={`${run.tool_name}-${run.finished_at}-${index}`} style={toolRowStyle}>
                                    <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                        {run.tool_name}.{run.capability}
                                    </div>
                                    <div style={mutedTextStyle}>
                                        {run.status} - {run.finished_at || run.started_at || 'time unknown'}
                                    </div>
                                    {run.evidence_status && (
                                        <div style={mutedTextStyle}>Evidence: {run.evidence_status}</div>
                                    )}
                                    {compactRunSummary(run, metrics) && (
                                        <div
                                            style={{
                                                ...mutedTextStyle,
                                                color: run.tool_name === 'security_suite' ? riskColor(metrics?.risk_level) : mutedTextStyle.color,
                                                fontWeight: run.tool_name === 'security_suite' ? 700 : 400,
                                            }}
                                        >
                                            {compactRunSummary(run, metrics)}
                                        </div>
                                    )}
                                    {Array.isArray(metrics?.findings) && metrics.findings.length > 0 && (
                                        <div style={mutedTextStyle}>
                                            Top finding: {metrics.findings[0].message || metrics.findings[0].category}
                                        </div>
                                    )}
                                    {Array.isArray(metrics?.detected_features) && (
                                        <div style={mutedTextStyle}>
                                            Features: {metrics.detected_features.slice(0, 3).join(', ')}
                                        </div>
                                    )}
                                    {run.artifact_path && (
                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginTop: '8px' }}>
                                            <div style={{ ...mutedTextStyle, wordBreak: 'break-all' }}>
                                                {run.artifact_path}
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => viewArtifact(run)}
                                                disabled={artifactLoading}
                                                style={{
                                                    border: '1px solid rgba(0, 212, 255, 0.22)',
                                                    borderRadius: '8px',
                                                    padding: '5px 8px',
                                                    background: 'rgba(7, 12, 28, 0.86)',
                                                    color: '#dff9ff',
                                                    cursor: 'pointer',
                                                    fontSize: '0.72rem',
                                                    whiteSpace: 'nowrap',
                                                }}
                                            >
                                                View
                                            </button>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                        {recentRuns.length === 0 && (
                            <div style={mutedTextStyle}>No tool runs recorded yet.</div>
                        )}
                    </div>
                    {selectedArtifact && (
                        <div style={{ ...toolRowStyle, marginTop: '10px' }}>
                            <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                Artifact Preview
                            </div>
                            <div style={{ ...mutedTextStyle, wordBreak: 'break-all', marginBottom: '8px' }}>
                                {selectedArtifact.artifact_path}
                            </div>
                            {selectedArtifact.artifact?.bayesian_assessment && (
                                <div style={mutedTextStyle}>
                                    Bayesian probability: {selectedArtifact.artifact.bayesian_assessment.probability ?? 'unknown'}
                                    {' '}confidence: {selectedArtifact.artifact.bayesian_assessment.confidence ?? 'unknown'}
                                </div>
                            )}
                            {Array.isArray(selectedArtifact.artifact?.bayesian_evidence) && selectedArtifact.artifact.bayesian_evidence.length > 0 && (
                                <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                                    Evidence:{' '}
                                    {selectedArtifact.artifact.bayesian_evidence.map((item) => {
                                        const state = item.status === 'skipped'
                                            ? `skipped: ${item.reason || 'policy'}`
                                            : (item.passed ? 'passed' : 'failed');
                                        return `${item.factor}: ${state}`;
                                    }).join('; ')}
                                </div>
                            )}
                            {(selectedArtifact.artifact?.target_path || selectedArtifact.artifact?.metrics?.target_path) && (
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '10px' }}>
                                    <button
                                        type="button"
                                        onClick={prepareQuarantineFromArtifact}
                                        disabled={commandLoading}
                                        style={{ ...runButtonStyle, opacity: commandLoading ? 0.55 : 1 }}
                                    >
                                        <Shield style={{ width: '14px', height: '14px' }} />
                                        Prepare Quarantine
                                    </button>
                                    {(selectedArtifact.artifact?.metadata_path || selectedArtifact.artifact?.metrics?.metadata_path) && (
                                        <button
                                            type="button"
                                            onClick={prepareRestoreFromArtifact}
                                            disabled={commandLoading}
                                            style={{ ...runButtonStyle, opacity: commandLoading ? 0.55 : 1 }}
                                        >
                                            <RotateCcw style={{ width: '14px', height: '14px' }} />
                                            Prepare Restore
                                        </button>
                                    )}
                                </div>
                            )}
                            {selectedArtifact.artifact?.metrics && (
                                <div style={mutedTextStyle}>
                                    Metrics: {JSON.stringify(selectedArtifact.artifact.metrics)}
                                </div>
                            )}
                            {Array.isArray(selectedArtifact.artifact?.recommended_actions || selectedArtifact.artifact?.metrics?.recommended_actions) && (
                                <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                                    Recommended:{' '}
                                    {(selectedArtifact.artifact.recommended_actions || selectedArtifact.artifact.metrics.recommended_actions)
                                        .slice(0, 4)
                                        .join('; ')}
                                </div>
                            )}
                            {Array.isArray(selectedArtifact.artifact?.detectors || selectedArtifact.artifact?.metrics?.detectors) && (
                                <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                                    Detectors:{' '}
                                    {(selectedArtifact.artifact.detectors || selectedArtifact.artifact.metrics.detectors)
                                        .slice(0, 5)
                                        .map((item) => `${item.name || 'detector'}:${item.status || 'unknown'}`)
                                        .join('; ')}
                                </div>
                            )}
                            {(selectedArtifact.artifact?.risk_level || selectedArtifact.artifact?.metrics?.risk_level) && (
                                <div
                                    style={{
                                        marginTop: '8px',
                                        color: riskColor(selectedArtifact.artifact.risk_level || selectedArtifact.artifact.metrics?.risk_level),
                                        fontWeight: 800,
                                        fontSize: '0.84rem',
                                    }}
                                >
                                    Risk: {selectedArtifact.artifact.risk_level || selectedArtifact.artifact.metrics?.risk_level}
                                    {' '}score {selectedArtifact.artifact.risk_score ?? selectedArtifact.artifact.metrics?.risk_score ?? 'unknown'}/100
                                </div>
                            )}
                            {Array.isArray(selectedArtifact.artifact?.findings || selectedArtifact.artifact?.metrics?.findings) && (
                                <div style={{ ...mutedTextStyle, marginTop: '6px' }}>
                                    Findings:{' '}
                                    {(selectedArtifact.artifact.findings || selectedArtifact.artifact.metrics.findings)
                                        .slice(0, 4)
                                        .map((item) => `${item.severity || 'info'}: ${item.message || item.category}`)
                                        .join('; ')}
                                </div>
                            )}
                            <pre
                                style={{
                                    marginTop: '8px',
                                    maxHeight: '220px',
                                    overflow: 'auto',
                                    whiteSpace: 'pre-wrap',
                                    color: '#dff9ff',
                                    background: 'rgba(0, 0, 0, 0.22)',
                                    border: '1px solid rgba(0, 212, 255, 0.12)',
                                    borderRadius: '8px',
                                    padding: '10px',
                                    fontSize: '0.72rem',
                                    lineHeight: 1.45,
                                }}
                            >
                                {selectedArtifact.artifact
                                    ? JSON.stringify(selectedArtifact.artifact, null, 2)
                                    : selectedArtifact.raw_text}
                            </pre>
                        </div>
                    )}
                </div>

                <div style={sectionStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Terminal style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Execution Plans</span>
                    </div>
                    <div style={{ display: 'grid', gap: '8px' }}>
                        {recentPlans.map((plan) => (
                            <div key={plan.id} style={toolRowStyle}>
                                <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                    {plan.intent}
                                </div>
                                <div style={mutedTextStyle}>
                                    {plan.status} - {plan.source} - {plan.steps?.length || 0} steps
                                </div>
                                {plan.result_summary && (
                                    <div style={{ ...mutedTextStyle, marginTop: '6px' }}>{plan.result_summary}</div>
                                )}
                                <button
                                    type="button"
                                    onClick={() => setSelectedPlan(plan)}
                                    style={{
                                        border: '1px solid rgba(0, 212, 255, 0.22)',
                                        borderRadius: '8px',
                                        padding: '5px 8px',
                                        background: 'rgba(7, 12, 28, 0.86)',
                                        color: '#dff9ff',
                                        cursor: 'pointer',
                                        fontSize: '0.72rem',
                                        marginTop: '8px',
                                    }}
                                >
                                    Details
                                </button>
                            </div>
                        ))}
                        {recentPlans.length === 0 && (
                            <div style={mutedTextStyle}>No runtime plans recorded yet.</div>
                        )}
                    </div>
                    {selectedPlan && (
                        <div style={{ ...toolRowStyle, marginTop: '10px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                                <div>
                                    <div style={{ color: '#e0f2fe', fontSize: '0.82rem', fontWeight: 700 }}>
                                        Plan Detail
                                    </div>
                                    <div style={mutedTextStyle}>{selectedPlan.id}</div>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => setSelectedPlan(null)}
                                    title="Close plan detail"
                                    style={{ ...iconButtonStyle, width: '28px', height: '28px' }}
                                >
                                    <X style={{ width: '14px', height: '14px' }} />
                                </button>
                            </div>
                            <div style={{ ...mutedTextStyle, marginTop: '8px' }}>
                                {selectedPlan.status} - {selectedPlan.source} - {selectedPlan.finished_at || selectedPlan.created_at}
                            </div>
                            {selectedPlan.error && (
                                <div style={{ ...mutedTextStyle, color: '#fecaca', marginTop: '6px' }}>
                                    Error: {selectedPlan.error}
                                </div>
                            )}
                            <div style={{ display: 'grid', gap: '8px', marginTop: '8px' }}>
                                {(selectedPlan.steps || []).map((step, index) => (
                                    <div key={`${step.name || step.tool || 'step'}-${index}`} style={{ borderTop: '1px solid rgba(0, 212, 255, 0.1)', paddingTop: '8px' }}>
                                        <div style={{ color: '#e0f2fe', fontSize: '0.78rem', fontWeight: 700 }}>
                                            {index + 1}. {step.name || step.tool || step.action || 'step'}
                                        </div>
                                        <div style={mutedTextStyle}>
                                            {step.status || 'planned'}{step.capability ? ` - ${step.capability}` : ''}
                                        </div>
                                        {(step.summary || step.result_summary || step.error) && (
                                            <div style={{ ...mutedTextStyle, marginTop: '4px' }}>
                                                {step.summary || step.result_summary || step.error}
                                            </div>
                                        )}
                                        {(step.input || step.params) && (
                                            <pre
                                                style={{
                                                    marginTop: '6px',
                                                    whiteSpace: 'pre-wrap',
                                                    color: '#dff9ff',
                                                    background: 'rgba(0, 0, 0, 0.18)',
                                                    borderRadius: '8px',
                                                    padding: '8px',
                                                    fontSize: '0.7rem',
                                                    lineHeight: 1.4,
                                                }}
                                            >
                                                {JSON.stringify(step.input || step.params, null, 2)}
                                            </pre>
                                        )}
                                    </div>
                                ))}
                                {(!selectedPlan.steps || selectedPlan.steps.length === 0) && (
                                    <div style={mutedTextStyle}>No plan steps were recorded.</div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div style={{ ...sectionStyle, borderBottom: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                        <Terminal style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                        <span style={{ color: '#ecfeff', fontWeight: 700, fontSize: '0.86rem' }}>Safe Commands</span>
                    </div>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {safeCommands.map((command) => (
                            <button
                                key={command}
                                type="button"
                                onClick={() => runCommand(command)}
                                disabled={commandLoading}
                                style={{
                                    border: '1px solid rgba(0, 212, 255, 0.22)',
                                    borderRadius: '8px',
                                    padding: '7px 10px',
                                    background: 'rgba(13, 24, 48, 0.8)',
                                    color: '#dff9ff',
                                    cursor: 'pointer',
                                    fontSize: '0.78rem',
                                }}
                            >
                                {command}
                            </button>
                        ))}
                    </div>
                    {commandOutput && (
                        <pre
                            style={{
                                marginTop: '10px',
                                whiteSpace: 'pre-wrap',
                                color: '#dff9ff',
                                background: 'rgba(0, 0, 0, 0.22)',
                                border: '1px solid rgba(0, 212, 255, 0.12)',
                                borderRadius: '8px',
                                padding: '10px',
                                fontSize: '0.76rem',
                                lineHeight: 1.45,
                            }}
                        >
                            {commandOutput}
                        </pre>
                    )}
                </div>
            </div>
        </aside>
    );
}
