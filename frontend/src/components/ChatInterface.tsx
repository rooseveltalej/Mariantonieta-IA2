import React from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';

const ChatInterface: React.FC = () => {
  // Datos de ejemplo para poblar el chat
  const mockMessages = [
    { sender: 'ai' as const, text: "Hello! I recognize you. How can I help you today?", timestamp: "04:49 p.m." },
    { sender: 'user' as const, text: "[Facial Recognition Activated]", timestamp: "04:49 p.m." },
    { sender: 'ai' as const, text: "I'm here to help! What would you like to know?", timestamp: "04:49 p.m." },
    { sender: 'user' as const, text: "hola", timestamp: "04:50 p.m." },
  ];

  return (
    <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
    }}>
      {/* Área de Mensajes */}
      <div style={{
          flexGrow: 1,
          overflowY: 'auto',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
      }}>
        {mockMessages.map((msg, index) => (
          <ChatMessage
            key={index}
            sender={msg.sender}
            text={msg.text}
            timestamp={msg.timestamp}
          />
        ))}
      </div>
      
      {/* Área de Entrada */}
      <ChatInput />
    </div>
  );
};

export default ChatInterface;