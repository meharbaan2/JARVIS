import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function AudioWaveVisualizer({ isActive, assistantName = 'JARVIS' }) {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const barsRef = useRef([]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        // Set canvas size
        const resizeCanvas = () => {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            canvas.style.width = `${rect.width}px`;
            canvas.style.height = `${rect.height}px`;
        };

        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Initialize bars
        const barCount = 60;
        if (barsRef.current.length === 0) {
            barsRef.current = Array.from({ length: barCount }, (_, i) => ({
                height: 0,
                targetHeight: Math.random() * 0.5 + 0.2,
                speed: Math.random() * 0.05 + 0.02,
                phase: (i / barCount) * Math.PI * 2
            }));
        }

        const animate = (time) => {
            const rect = canvas.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            // Clear canvas
            ctx.clearRect(0, 0, width, height);

            const barWidth = width / barCount;
            const centerY = height / 2;

            barsRef.current.forEach((bar, i) => {
                if (isActive) {
                    // Animate towards target
                    bar.height += (bar.targetHeight - bar.height) * bar.speed;

                    // Change target occasionally
                    if (Math.random() < 0.02) {
                        bar.targetHeight = Math.random() * 0.8 + 0.2;
                    }

                    // Add wave effect
                    const wave = Math.sin(time * 0.003 + bar.phase) * 0.2;
                    const finalHeight = (bar.height + wave) * height * 0.4;

                    const x = i * barWidth + barWidth / 2;

                    // Create gradient
                    const gradient = ctx.createLinearGradient(x, centerY - finalHeight, x, centerY + finalHeight);
                    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.8)');
                    gradient.addColorStop(0.5, 'rgba(0, 153, 255, 1)');
                    gradient.addColorStop(1, 'rgba(0, 212, 255, 0.8)');

                    ctx.fillStyle = gradient;

                    // Draw bar with rounded ends
                    const barDrawWidth = barWidth * 0.6;
                    const radius = barDrawWidth / 2;

                    ctx.beginPath();
                    ctx.roundRect(
                        x - barDrawWidth / 2,
                        centerY - finalHeight,
                        barDrawWidth,
                        finalHeight * 2,
                        radius
                    );
                    ctx.fill();

                    // Add glow
                    ctx.shadowBlur = 15;
                    ctx.shadowColor = 'rgba(0, 212, 255, 0.5)';
                    ctx.fill();
                    ctx.shadowBlur = 0;

                } else {
                    // Fade out
                    bar.height *= 0.95;
                    if (bar.height > 0.01) {
                        const finalHeight = bar.height * height * 0.4;
                        const x = i * barWidth + barWidth / 2;
                        const barDrawWidth = barWidth * 0.6;
                        const radius = barDrawWidth / 2;

                        ctx.fillStyle = `rgba(0, 212, 255, ${bar.height})`;
                        ctx.beginPath();
                        ctx.roundRect(
                            x - barDrawWidth / 2,
                            centerY - finalHeight,
                            barDrawWidth,
                            finalHeight * 2,
                            radius
                        );
                        ctx.fill();
                    }
                }
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isActive]);

    // Inline styles for the container
    const containerStyle = {
        position: 'relative',
        width: '100%'
    };

    const glassContainerStyle = {
        position: 'relative',
        backdropFilter: 'blur(24px)',
        background: 'linear-gradient(to right, rgba(0, 212, 255, 0.1), rgba(59, 130, 246, 0.1), rgba(0, 212, 255, 0.1))',
        borderRadius: '16px',
        border: '1px solid rgba(0, 212, 255, 0.2)',
        boxShadow: '0 8px 32px rgba(0, 212, 255, 0.15)',
        padding: '4px',
        overflow: 'hidden'
    };

    const innerGlowStyle = {
        position: 'absolute',
        inset: 0,
        background: 'linear-gradient(to right, rgba(0, 212, 255, 0), rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0))',
        animation: 'pulse 2s infinite'
    };

    const contentStyle = {
        position: 'relative',
        background: 'rgba(10, 10, 26, 0.6)',
        borderRadius: '12px',
        padding: '24px'
    };

    const statusBadgeStyle = {
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 16px',
        borderRadius: '9999px',
        background: 'rgba(0, 212, 255, 0.1)',
        border: '1px solid rgba(0, 212, 255, 0.3)'
    };

    const pulseDotStyle = {
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        background: '#22d3ee',
        animation: 'pulse 2s infinite',
        boxShadow: '0 0 10px rgba(0, 212, 255, 0.8)'
    };

    const statusTextStyle = {
        color: '#22d3ee',
        fontSize: '0.875rem',
        fontWeight: 500,
        letterSpacing: '0.05em'
    };

    const canvasStyle = {
        width: '100%',
        height: '128px',
        borderRadius: '8px'
    };

    const reflectionStyle = {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '33%',
        background: 'linear-gradient(to top, rgba(0, 212, 255, 0.05), transparent)',
        pointerEvents: 'none'
    };

    const bottomGlowStyle = {
        position: 'absolute',
        bottom: '-16px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '66%',
        height: '32px',
        background: 'rgba(0, 212, 255, 0.2)',
        filter: 'blur(32px)'
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            style={containerStyle}
        >
            {/* Glass container */}
            <div style={glassContainerStyle}>
                <div style={innerGlowStyle}></div>

                <div style={contentStyle}>
                    <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                        <div style={statusBadgeStyle}>
                            <div style={pulseDotStyle}></div>
                            <span style={statusTextStyle}>
                                {assistantName} SPEAKING
                            </span>
                        </div>
                    </div>

                    <canvas
                        ref={canvasRef}
                        style={canvasStyle}
                    />

                    {/* Reflection effect */}
                    <div style={reflectionStyle}></div>
                </div>
            </div>

            {/* Bottom glow */}
            <div style={bottomGlowStyle}></div>
        </motion.div>
    );
}