import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Cpu } from 'lucide-react';

function looksLikeToolResult(text) {
    const lowered = String(text || '').toLowerCase();
    return (
        lowered.includes('artifact:')
        || lowered.includes('evidence:')
        || lowered.includes('verdict:')
        || lowered.includes('parameters:')
        || lowered.includes('security_suite')
        || lowered.includes('ofdmsim')
        || lowered.includes('cadconverter')
        || lowered.includes('folderguardian')
        || lowered.includes('bayesian')
    );
}

function parseToolResultText(text) {
    const lines = String(text || '').split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const parsed = {
        title: lines[0] || 'Runtime Result',
        rows: [],
        evidence: [],
        notes: [],
        artifact: '',
    };
    let collectingEvidence = false;
    const pushRow = (label, value) => {
        if (value) parsed.rows.push({ label, value });
    };
    for (const line of lines.slice(1)) {
        if (/^artifact:/i.test(line)) {
            parsed.artifact = line.replace(/^artifact:\s*/i, '').trim();
            collectingEvidence = false;
            continue;
        }
        if (/^evidence:/i.test(line)) {
            const value = line.replace(/^evidence:\s*/i, '').trim();
            if (value) parsed.evidence.push(value);
            collectingEvidence = true;
            continue;
        }
        if (/^bayesian .*updated/i.test(line)) {
            parsed.rows.push({ label: 'Bayesian', value: line });
            collectingEvidence = false;
            continue;
        }
        if (line.startsWith('- ')) {
            const value = line.slice(2).trim();
            if (collectingEvidence || /^(low|medium|high|critical|info|success|failed|skipped)/i.test(value)) {
                parsed.evidence.push(value);
            } else {
                parsed.notes.push(value);
            }
            continue;
        }
        collectingEvidence = false;
        const pair = line.match(/^([A-Za-z][A-Za-z0-9 /_-]{1,28}):\s*(.+)$/);
        if (pair) {
            pushRow(pair[1], pair[2]);
            continue;
        }
        if (/^(ber|baseline ber|emi ber|clipped ber|processed|manifests|dependencies found)\b/i.test(line)) {
            parsed.rows.push({ label: 'Result', value: line });
            continue;
        }
        parsed.notes.push(line);
    }
    if (!parsed.rows.length && !parsed.evidence.length && parsed.notes.length > 3) {
        parsed.rows.push({ label: 'Summary', value: parsed.notes.shift() });
    }
    return parsed;
}

export default function MessageBubble({ message }) {
    const isJarvis = message.sender === 'jarvis';
    const [isHovered, setIsHovered] = useState(false);
    const MotionDiv = motion.div;

    // Outer container styles
    const outerContainerStyle = {
        display: 'flex',
        justifyContent: isJarvis ? 'flex-start' : 'flex-end'
    };

    // Inner container styles
    const innerContainerStyle = {
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
        maxWidth: '80%',
        flexDirection: isJarvis ? 'row' : 'row-reverse'
    };

    const avatarContainerStyle = {
        flexShrink: 0,
        position: 'relative'
    };

    const avatarStyle = {
        width: '40px',
        height: '40px',
        borderRadius: '12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: isJarvis 
            ? 'linear-gradient(135deg, #06b6d4, #2563eb)'
            : 'linear-gradient(135deg, #a855f7, #ec4899)',
        boxShadow: isJarvis
            ? '0 0 20px rgba(0, 212, 255, 0.4)'
            : '0 0 20px rgba(168, 85, 247, 0.4)'
    };

    const avatarGlowStyle = {
        position: 'absolute',
        inset: 0,
        borderRadius: '12px',
        filter: 'blur(8px)',
        opacity: 0.4,
        background: isJarvis ? '#22d3ee' : '#a855f7',
        animation: isJarvis ? 'pulse 2s infinite' : 'none'
    };

    const messageContentStyle = {
        flex: 1
    };

    const bubbleWrapperStyle = {
        position: 'relative'
    };

    const bubbleStyle = {
        position: 'relative',
        backdropFilter: 'blur(24px)',
        borderRadius: '16px',
        border: '1px solid',
        padding: '16px',
        background: isJarvis
            ? 'linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(59, 130, 246, 0.1), rgba(0, 212, 255, 0.05))'
            : 'linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.1), rgba(168, 85, 247, 0.05))',
        borderColor: isJarvis
            ? 'rgba(0, 212, 255, 0.2)'
            : 'rgba(168, 85, 247, 0.2)',
        boxShadow: isJarvis
            ? '0 8px 32px rgba(0, 212, 255, 0.1)'
            : '0 8px 32px rgba(168, 85, 247, 0.1)'
    };

    const innerGlowStyle = {
        position: 'absolute',
        inset: 0,
        borderRadius: '16px',
        background: isJarvis
            ? 'linear-gradient(to right, rgba(0, 212, 255, 0), rgba(0, 212, 255, 0.05), rgba(0, 212, 255, 0))'
            : 'linear-gradient(to right, rgba(168, 85, 247, 0), rgba(168, 85, 247, 0.05), rgba(168, 85, 247, 0))'
    };

    const contentStyle = {
        position: 'relative'
    };

    const nameStyle = {
        fontSize: '0.75rem',
        color: 'rgba(0, 212, 255, 0.6)',
        fontWeight: 500,
        letterSpacing: '0.05em',
        marginBottom: '8px'
    };

    const textStyle = {
        fontSize: '0.875rem',
        lineHeight: '1.5',
        color: isJarvis ? '#e0f2fe' : '#fae8ff'
    };

    const timestampStyle = {
        fontSize: '0.75rem',
        marginTop: '8px',
        color: isJarvis ? 'rgba(0, 212, 255, 0.4)' : 'rgba(168, 85, 247, 0.4)'
    };
    const mutedParagraphStyle = {
        color: 'rgba(186, 230, 253, 0.68)',
        fontSize: '0.82rem',
        lineHeight: 1.45,
    };
    const sourceBadgeStyle = {
        display: 'inline-flex',
        alignItems: 'center',
        border: '1px solid rgba(34, 211, 238, 0.22)',
        borderRadius: '999px',
        padding: '2px 8px',
        color: 'rgba(165, 243, 252, 0.86)',
        background: 'rgba(8, 47, 73, 0.34)',
        fontSize: '0.68rem',
        fontWeight: 700,
        letterSpacing: '0.04em',
        marginLeft: '8px',
        verticalAlign: 'middle'
    };
    const detailRowStyle = {
        display: 'grid',
        gridTemplateColumns: '86px minmax(0, 1fr)',
        gap: '8px',
        padding: '5px 0',
        borderTop: '1px solid rgba(125, 211, 252, 0.08)',
    };
    const detailLabelStyle = {
        color: 'rgba(165, 243, 252, 0.72)',
        fontSize: '0.76rem',
        fontWeight: 700,
    };
    const detailValueStyle = {
        color: isJarvis ? '#e0f2fe' : '#fae8ff',
        fontSize: '0.84rem',
        wordBreak: 'break-word',
    };
    const statusChipStyle = (value) => {
        const lowered = String(value || '').toLowerCase();
        let color = '#7dd3fc';
        let background = 'rgba(14, 116, 144, 0.2)';
        let border = 'rgba(125, 211, 252, 0.24)';
        if (/\bcritical\b|\bhigh risk\b|\bfailed\b|\bagainst\b/.test(lowered)) {
            color = '#fecaca';
            background = 'rgba(127, 29, 29, 0.35)';
            border = 'rgba(248, 113, 113, 0.35)';
        } else if (/\belevated\b|\bmedium\b|\bskipped\b|\bwarning\b/.test(lowered)) {
            color = '#fde68a';
            background = 'rgba(113, 63, 18, 0.32)';
            border = 'rgba(251, 191, 36, 0.3)';
        } else if (/\blow\b|\bsuccess\b|\bcomplete\b|\badded\b|\bupdated\b/.test(lowered)) {
            color = '#bbf7d0';
            background = 'rgba(20, 83, 45, 0.28)';
            border = 'rgba(74, 222, 128, 0.28)';
        }
        return {
            display: 'inline-flex',
            alignItems: 'center',
            border: `1px solid ${border}`,
            borderRadius: '999px',
            padding: '2px 8px',
            color,
            background,
            fontSize: '0.76rem',
            fontWeight: 700,
            lineHeight: 1.5,
        };
    };
    const evidenceStyle = {
        margin: '8px 0 0',
        paddingLeft: '18px',
        color: isJarvis ? '#dff9ff' : '#fae8ff',
        fontSize: '0.84rem',
        lineHeight: 1.45,
    };
    const artifactStyle = {
        marginTop: '8px',
        padding: '7px 8px',
        borderRadius: '8px',
        background: 'rgba(2, 6, 23, 0.35)',
        color: 'rgba(186, 230, 253, 0.62)',
        fontSize: '0.72rem',
        wordBreak: 'break-all',
    };

    const shineEffectStyle = {
        position: 'absolute',
        inset: 0,
        borderRadius: '16px',
        background: isHovered 

            ? 'linear-gradient(to right, transparent, rgba(255, 255, 255, 0.05), transparent)'

            : 'linear-gradient(to right, transparent, rgba(255, 255, 255, 0), transparent)',
        transition: 'all 0.5s',
        pointerEvents: 'none'
    };

    const bottomGlowStyle = {
        position: 'absolute',
        bottom: '-8px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '75%',
        height: '16px',
        filter: 'blur(16px)',
        opacity: 0.3,
        background: isJarvis ? '#22d3ee' : '#a855f7'
    };

    const renderMessageContent = () => {
        const text = String(message.text || '');
        if (!isJarvis || !looksLikeToolResult(text)) {
            return <p style={textStyle}>{text}</p>;
        }
        const parsed = parseToolResultText(text);
        return (
            <div style={textStyle}>
                <div style={{ color: '#ecfeff', fontWeight: 700, marginBottom: '8px' }}>
                    {parsed.title}
                </div>
                {parsed.rows.map((row, index) => {
                    const chipLike = /verdict|risk|status|bayesian|evidence added/i.test(row.label);
                    return (
                        <div key={`${row.label}-${index}`} style={detailRowStyle}>
                            <div style={detailLabelStyle}>{row.label}</div>
                            <div style={detailValueStyle}>
                                {chipLike ? <span style={statusChipStyle(row.value)}>{row.value}</span> : row.value}
                            </div>
                        </div>
                    );
                })}
                {parsed.evidence.length > 0 && (
                    <ul style={evidenceStyle}>
                        {parsed.evidence.map((item, index) => (
                            <li key={`${item}-${index}`}>{item}</li>
                        ))}
                    </ul>
                )}
                {parsed.notes.map((note, index) => (
                    <div key={`${note}-${index}`} style={{ ...mutedParagraphStyle, marginTop: '8px' }}>
                        {note}
                    </div>
                ))}
                {parsed.artifact && (
                    <div style={artifactStyle}>
                        Artifact: {parsed.artifact}
                    </div>
                )}
            </div>
        );
    };

    return (
        <MotionDiv
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.3 }}
            style={outerContainerStyle}
        >
            {/* Inner flex container with max-width */}
            <div style={innerContainerStyle}>
                {/* Avatar stays with message now */}
                <div style={avatarContainerStyle}>
                    <div style={avatarStyle}>
                        {isJarvis ? (
                            <Cpu style={{ width: '20px', height: '20px', color: 'white' }} />
                        ) : (
                            <User style={{ width: '20px', height: '20px', color: 'white' }} />
                        )}
                    </div>
                    <div style={avatarGlowStyle}></div>
                </div>

                {/* Message bubble */}
                <div style={messageContentStyle}>
                    <div style={bubbleWrapperStyle}

                        onMouseEnter={() => setIsHovered(true)}

                        onMouseLeave={() => setIsHovered(false)}

                    >
                        {/* Glass morphism bubble */}
                        <div style={bubbleStyle}>
                            {/* Inner glow */}
                            <div style={innerGlowStyle}></div>

                            {/* Content */}
                            <div style={contentStyle}>
                                {isJarvis && (
                                    <div style={nameStyle}>
                                        J.A.R.V.I.S
                                        {message.source && (
                                            <span style={sourceBadgeStyle}>
                                                {message.source}
                                            </span>
                                        )}
                                    </div>
                                )}
                                {renderMessageContent()}
                                <div style={timestampStyle}>
                                    {new Date(message.timestamp).toLocaleTimeString([], {
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    })}
                                </div>
                            </div>

                            {/* Shine effect on hover */}
                            <div style={shineEffectStyle}></div>
                        </div>

                        {/* Bottom glow */}
                        <div style={bottomGlowStyle}></div>
                    </div>
                </div>
            </div>
        </MotionDiv>
    );
}
