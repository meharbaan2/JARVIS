import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Cpu } from 'lucide-react';

export default function MessageBubble({ message }) {
    const isJarvis = message.sender === 'jarvis';
    const [isHovered, setIsHovered] = useState(false);

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

    return (
        <motion.div
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
                                    </div>
                                )}
                                <p style={textStyle}>
                                    {message.text}
                                </p>
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
        </motion.div>
    );
}