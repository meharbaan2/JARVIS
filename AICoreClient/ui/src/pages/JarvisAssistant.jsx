import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Languages, Square } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import AudioWaveVisualizer from '../components/AudioWaveVisualizer';
import MessageBubble from '../components/MessageBubble';

export default function JarvisAssistant() {
    const [language, setLanguage] = useState('english');
    const [messages, setMessages] = useState([
        { sender: 'jarvis', text: 'Good evening, sir. J.A.R.V.I.S at your service.', timestamp: new Date() }
    ]);
    const [inputText, setInputText] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const messagesEndRef = useRef(null);
    const messagesContainerRef = useRef(null);

    const assistantName = language === 'punjabi' ? 'FRIDAY' : 'JARVIS';
    const assistantFullName = language === 'punjabi'
        ? 'Female Replacement Intelligent Digital Assistant Youth'
        : 'Just A Rather Very Intelligent System';

    // --- SPEECH INTERRUPTION FUNCTION ---
    const interruptSpeech = async () => {
        if (isSpeaking) {
            console.log('[UI] Interrupting current speech');
            try {
                // Immediately hide the wave visualizer
                setIsSpeaking(false);

                // Call the interrupt function in Electron
                if (window.electronAPI?.interruptSpeech) {
                    await window.electronAPI.interruptSpeech();
                    console.log('[UI] Speech interrupt signal sent');
                } else {
                    console.warn('[UI] interruptSpeech API not available');
                }
            } catch (error) {
                console.error('[UI] Failed to interrupt speech:', error);
            }
        }
    };

    // --- INTERRUPT SPEECH WHEN USER TYPES ---
    useEffect(() => {
        if (inputText.trim() && isSpeaking) {
            console.log('[UI] User started typing - interrupting speech');
            interruptSpeech();
        }
    }, [inputText]); // This runs whenever inputText changes


    // --- SPEECH EVENT HANDLER ---
    useEffect(() => {
        if (!window.electronAPI?.onSpeechEvent) {
            console.log('Speech events not available');
            return;
        }

        const handleSpeechEvent = (data) => {
            console.log('[UI] Speech event received:', data);

            if (data.type === 'recording_start') {
                console.log('[UI] User recording started');
                setIsRecording(true); // Show recording state
            }
            else if (data.type === 'recording_end') {
                console.log('[UI] User recording ended');
                setIsRecording(false); // Hide recording state
            }
            else if (data.type === 'jarvis_speak_start') {
                console.log('[UI] JARVIS speaking started - show wave');
                setIsSpeaking(true); // Show speaking visualizer
            }
            else if (data.type === 'jarvis_speak_end') {
                console.log('[UI] JARVIS speaking ended - hide wave');
                setIsSpeaking(false); // Hide speaking visualizer
            }
            else if (data.type === 'recognized' && data.text) {
                console.log('[UI] Recognized speech:', data.text);

                // Interrupt any ongoing speech before processing new voice input
                interruptSpeech();

                // Add the recognized text as a user message
                const userMessage = { sender: 'user', text: data.text, timestamp: new Date() };
                setMessages(prev => [...prev, userMessage]);

                // Send recognized text to AI backend
                window.electronAPI.sendMessage(data.text)
                    .then(response => {
                        const jarvisMessage = { sender: 'jarvis', text: response, timestamp: new Date() };
                        setMessages(prev => [...prev, jarvisMessage]);
                    })
                    .catch(err => {
                        console.error('[AI RESPONSE ERROR]', err);
                        const errorMessage = {
                            sender: 'jarvis',
                            text: 'Error getting response.',
                            timestamp: new Date()
                        };
                        setMessages(prev => [...prev, errorMessage]);
                    });
            }
        };

        window.electronAPI.onSpeechEvent(handleSpeechEvent);

        return () => {
            // optional: cleanup handler
            if (window.electronAPI?.removeAllListeners) {
                window.electronAPI.removeAllListeners('speech-event');
            }
        };
    }, []);


    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({
            behavior: 'smooth',
            block: 'end',
            inline: 'nearest'
        });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleLanguageChange = (newLanguage) => {
        setLanguage(newLanguage);
        const systemMessage = {
            sender: 'jarvis',
            text: newLanguage === 'punjabi'
                ? 'ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ F.R.I.D.A.Y. ਹਾਂ। ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦੀ ਹਾਂ?'
                : 'Language switched to English. J.A.R.V.I.S at your service.',
            timestamp: new Date()
        };
        setMessages(prev => [...prev, systemMessage]);
    };

    const handleSend = async () => {
        if (!inputText.trim()) return;

        // Interrupt any ongoing speech first
        await interruptSpeech();

        const userMessage = { sender: 'user', text: inputText, timestamp: new Date() };
        setMessages(prev => [...prev, userMessage]);
        setInputText('');

        try {
            console.log('[UI] Sending message to Electron:', inputText);

            if (!window.electronAPI || !window.electronAPI.sendMessage) {
                throw new Error('Electron API not available');
            }

            const responseText = await window.electronAPI.sendMessage(inputText);
            console.log('[UI] Received response from Electron:', responseText);

            const jarvisMessage = {
                sender: 'jarvis',
                text: responseText,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, jarvisMessage]);

        } catch (error) {
            console.error('[UI ERROR] Request failed:', error);
            const errorMessage = {
                sender: 'jarvis',
                text: `Connection error: ${error.message}`,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        }
    };

    const handleVoiceInput = async () => {
        if (isListening) {
            setIsListening(false);
            return;
        }

        // Interrupt any ongoing speech first
        await interruptSpeech();

        setIsListening(true);

        const userMessage = { sender: 'user', text: '[Listening...]', timestamp: new Date() };
        setMessages(prev => [...prev, userMessage]);

        try {
            const lang = language === 'punjabi' ? 'punjabi' : 'english';
            console.log('[UI] Triggering voice input:', lang);

            // Ask Electron to trigger the VoiceBridge.exe process
            await window.electronAPI.triggerVoice(lang);

        } catch (err) {
            console.error('[VOICE INPUT ERROR]', err);
            setIsListening(false);
            setMessages(prev => [
                ...prev,
                { sender: 'jarvis', text: 'Voice capture failed.', timestamp: new Date() }
            ]);
        }
    };


    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Inline styles for the complex effects
    const containerStyle = {
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #050510 0%, #0a0a1a 50%, #0d0d24 100%)'
    };

    const backgroundGlowStyle = {
        position: 'absolute',
        top: '25%',
        left: '25%',
        width: '384px',
        height: '384px',
        background: 'rgba(0, 212, 255, 0.1)',
        borderRadius: '50%',
        filter: 'blur(120px)',
        animation: 'pulse 2s infinite'
    };

    const gridOverlayStyle = {
        position: 'absolute',
        inset: 0,
        backgroundImage: `
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px',
        maskImage: 'radial-gradient(ellipse 80% 50% at 50% 50%, black, transparent)'
    };

    const glassContainerStyle = {
        backdropFilter: 'blur(16px)',
        background: 'linear-gradient(to right, rgba(0, 212, 255, 0.1), rgba(59, 130, 246, 0.1), rgba(0, 212, 255, 0.1))',
        border: '1px solid rgba(0, 212, 255, 0.2)',
        borderRadius: '16px',
        boxShadow: '0 8px 32px rgba(0, 212, 255, 0.1)'
    };

    const gradientTextStyle = {
        background: 'linear-gradient(to right, #22d3ee, #3b82f6, #22d3ee)',
        backgroundSize: '200% 200%',
        animation: 'gradientShift 3s ease infinite',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent'
    };

    return (
        <div style={containerStyle}>
            {/* Animated background elements */}
            <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
                <div style={backgroundGlowStyle}></div>
                <div style={{
                    ...backgroundGlowStyle,
                    top: 'auto',
                    bottom: '25%',
                    left: 'auto',
                    right: '25%',
                    background: 'rgba(59, 130, 246, 0.1)',
                    animationDelay: '1s'
                }}></div>
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '600px',
                    height: '600px',
                    background: 'rgba(0, 212, 255, 0.05)',
                    borderRadius: '50%',
                    filter: 'blur(150px)'
                }}></div>
            </div>

            {/* Grid overlay */}
            <div style={gridOverlayStyle}></div>

            <div style={{ position: 'relative', zIndex: 10, display: 'flex', flexDirection: 'column', height: '100vh', padding: '2rem' }}>
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ marginBottom: '2rem' }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <div style={{ position: 'relative' }}>
                                <div style={{
                                    width: '64px',
                                    height: '64px',
                                    borderRadius: '50%',
                                    background: 'linear-gradient(135deg, #22d3ee, #3b82f6)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    boxShadow: '0 0 30px rgba(0, 212, 255, 0.5)'
                                }}>
                                    <div style={{
                                        width: '56px',
                                        height: '56px',
                                        borderRadius: '50%',
                                        background: '#0a0a1a',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                    }}>
                                        <span style={{
                                            fontSize: '1.5rem',
                                            fontWeight: 'bold',
                                            background: 'linear-gradient(to right, #22d3ee, #3b82f6)',
                                            WebkitBackgroundClip: 'text',
                                            WebkitTextFillColor: 'transparent'
                                        }}>
                                            {assistantName[0]}
                                        </span>
                                    </div>
                                </div>
                                <div style={{
                                    position: 'absolute',
                                    inset: 0,
                                    borderRadius: '50%',
                                    background: '#22d3ee',
                                    filter: 'blur(20px)',
                                    opacity: 0.3,
                                    animation: 'pulse 2s infinite'
                                }}></div>
                            </div>
                            <div>
                                <h1 style={gradientTextStyle}>
                                    {assistantName}
                                </h1>
                                <p style={{
                                    color: 'rgba(0, 212, 255, 0.6)',
                                    fontSize: '0.875rem',
                                    letterSpacing: '0.2em',
                                    marginTop: '0.25rem',
                                    textTransform: 'uppercase'
                                }}>
                                    {assistantFullName}
                                </p>
                            </div>
                        </div>

                        {/* Language Selector */}
                        <div style={{ position: 'relative' }}>
                            <div style={glassContainerStyle}>
                                <div style={{
                                    background: 'rgba(10, 10, 26, 0.6)',
                                    borderRadius: '12px',
                                    padding: '12px'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                        <Languages style={{ width: '16px', height: '16px', color: '#22d3ee' }} />
                                        <span style={{ color: '#22d3ee', fontSize: '0.75rem', fontWeight: 500, letterSpacing: '0.1em' }}>LANGUAGE</span>
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <button
                                            onClick={() => handleLanguageChange('english')}
                                            style={{
                                                padding: '8px 16px',
                                                borderRadius: '8px',
                                                fontSize: '0.875rem',
                                                fontWeight: 500,
                                                transition: 'all 0.3s',
                                                ...(language === 'english' ? {
                                                    background: 'linear-gradient(to right, #22d3ee, #3b82f6)',
                                                    color: 'white',
                                                    boxShadow: '0 0 20px rgba(0, 212, 255, 0.4)'
                                                } : {
                                                    background: 'rgba(13, 13, 36, 0.5)',
                                                    color: 'rgba(0, 212, 255, 0.6)',
                                                    border: '1px solid rgba(0, 212, 255, 0.2)'
                                                })
                                            }}
                                        >
                                            English
                                        </button>
                                        <button
                                            onClick={() => handleLanguageChange('punjabi')}
                                            style={{
                                                padding: '8px 16px',
                                                borderRadius: '8px',
                                                fontSize: '0.875rem',
                                                fontWeight: 500,
                                                transition: 'all 0.3s',
                                                ...(language === 'punjabi' ? {
                                                    background: 'linear-gradient(to right, #22d3ee, #3b82f6)',
                                                    color: 'white',
                                                    boxShadow: '0 0 20px rgba(0, 212, 255, 0.4)'
                                                } : {
                                                    background: 'rgba(13, 13, 36, 0.5)',
                                                    color: 'rgba(0, 212, 255, 0.6)',
                                                    border: '1px solid rgba(0, 212, 255, 0.2)'
                                                })
                                            }}
                                        >
                                            ਪੰਜਾਬੀ
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Audio Wave Visualizer - Fixed overlay that doesn't affect layout */}
                <AnimatePresence>
                    {isSpeaking && (
                        <motion.div
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                            style={{
                                position: 'fixed', // Changed from sticky to absolute
                                top: '140px', // Position below the header
                                left: '2rem',
                                right: '2rem',
                                zIndex: 1000, // Very high z-index to ensure it's above everything
                                pointerEvents: 'none' // Don't interfere with clicks
                            }}
                        >
                            <AudioWaveVisualizer isActive={isSpeaking} assistantName={assistantName} />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Messages Container */}
                <div
                    ref={messagesContainerRef}
                    style={{
                        flex: 1,
                        overflow: 'hidden',
                        marginBottom: '1.5rem',
                        position: 'relative',
                    }}
                >
                    <div
                        className="messages-container"
                        style={{
                            height: '100%',
                            overflowY: 'auto',
                            padding: '0 1rem 30px 1rem',
                        }}
                    >

                        {/* Custom scrollbar styles */}
                        <style>
                            {`
                                .messages-container::-webkit-scrollbar {
                                    width: 8px;
                                }
                                .messages-container::-webkit-scrollbar-track {
                                    background: rgba(0, 212, 255, 0.1);
                                    border-radius: 4px;
                                    margin: 4px 0;
                                }
                                .messages-container::-webkit-scrollbar-thumb {
                                    background: linear-gradient(to bottom, #22d3ee, #3b82f6);
                                    border-radius: 4px;
                                    border: 1px solid rgba(0, 212, 255, 0.3);
                                }
                                .messages-container::-webkit-scrollbar-thumb:hover {
                                    background: linear-gradient(to bottom, #0ea5e9, #2563eb);
                                }
                
                                /* For Firefox */
                                .messages-container {
                                    scrollbar-width: thin;
                                    scrollbar-color: #22d3ee rgba(0, 212, 255, 0.1);
                                }
                            `}
                        </style>

                        <AnimatePresence>
                            {messages.map((message, index) => (
                                <div key={index} style={{ marginBottom: '1rem' }}>
                                    <MessageBubble message={message} />
                                </div>
                            ))}
                        </AnimatePresence>
                        <div ref={messagesEndRef} style={{ height: '20px' }} />
                    </div>
                </div>

                {/* Input Area */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    style={{ position: 'relative' }}
                >
                    <div style={glassContainerStyle}>
                        <div style={{
                            position: 'absolute',
                            inset: 0,
                            background: 'linear-gradient(to right, rgba(0, 212, 255, 0), rgba(0, 212, 255, 0.1), rgba(0, 212, 255, 0))',
                            borderRadius: '16px',
                            filter: 'blur(20px)'
                        }}></div>

                        <div style={{
                            position: 'relative',
                            background: 'rgba(10, 10, 26, 0.6)',
                            borderRadius: '12px',
                            padding: '1rem'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.75rem' }}>
                                {/* Voice button */}
                                <button
                                    onClick={handleVoiceInput}
                                    style={{
                                        flexShrink: 0,
                                        width: '48px',
                                        height: '48px',
                                        borderRadius: '12px',
                                        transition: 'all 0.3s',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        position: 'relative',
                                        overflow: 'hidden',
                                        ...(isListening ? {
                                            background: 'linear-gradient(135deg, #ef4444, #ec4899)',
                                            boxShadow: '0 0 20px rgba(239, 68, 68, 0.5)'
                                        } : {
                                            background: 'linear-gradient(135deg, #22d3ee, #3b82f6)'
                                        })
                                    }}
                                >
                                    {isListening ? (
                                        <MicOff style={{ width: '20px', height: '20px', color: 'white', position: 'relative', zIndex: 10 }} />
                                    ) : (
                                        <Mic style={{ width: '20px', height: '20px', color: 'white', position: 'relative', zIndex: 10 }} />
                                    )}
                                    {isListening && (
                                        <div style={{
                                            position: 'absolute',
                                            inset: 0,
                                            background: '#f87171',
                                            borderRadius: '12px',
                                            animation: 'pulse 2s infinite',
                                            opacity: 0.2
                                        }}></div>
                                    )}
                                </button>

                                {/* Text input */}
                                <div style={{ flex: 1, position: 'relative' }}>
                                    <input
                                        type="text"
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        placeholder={language === 'punjabi' ? 'ਆਪਣਾ ਸਵਾਲ ਪੁੱਛੋ...' : 'Speak or type your command...'}
                                        style={{
                                            width: '100%',
                                            background: 'rgba(13, 13, 36, 0.5)',
                                            border: '1px solid rgba(0, 212, 255, 0.2)',
                                            borderRadius: '12px',
                                            padding: '12px 16px',
                                            color: '#e0f2fe',
                                            outline: 'none',
                                            transition: 'all 0.3s'
                                        }}
                                        onFocus={(e) => {
                                            e.target.style.borderColor = 'rgba(0, 212, 255, 0.5)';
                                            e.target.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.2)';
                                        }}
                                        onBlur={(e) => {
                                            e.target.style.borderColor = 'rgba(0, 212, 255, 0.2)';
                                            e.target.style.boxShadow = 'none';
                                        }}
                                    />
                                </div>

                                {/* Send button */}
                                <button
                                    onClick={handleSend}
                                    disabled={!inputText.trim()}
                                    style={{
                                        flexShrink: 0,
                                        width: '48px',
                                        height: '48px',
                                        borderRadius: '12px',
                                        transition: 'all 0.3s',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        position: 'relative',
                                        overflow: 'hidden',
                                        ...(inputText.trim() ? {
                                            background: 'linear-gradient(135deg, #22d3ee, #3b82f6)',
                                            cursor: 'pointer'
                                        } : {
                                            background: 'rgba(55, 65, 81, 0.5)',
                                            cursor: 'not-allowed',
                                            opacity: 0.5
                                        })
                                    }}
                                >
                                    <Send style={{ width: '20px', height: '20px', color: 'white', position: 'relative', zIndex: 10 }} />
                                </button>
                            </div>

                            {/* Status indicator */}
                            {isListening && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#f87171', fontSize: '0.875rem' }}
                                >
                                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f87171', animation: 'pulse 2s infinite' }}></div>
                                    {language === 'punjabi' ? 'ਸੁਣ ਰਿਹਾ ਹੈ...' : 'Listening...'}
                                </motion.div>
                            )}
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}