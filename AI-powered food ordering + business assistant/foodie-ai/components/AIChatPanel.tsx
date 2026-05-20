'use client';

import { useState, useRef, useEffect } from 'react';
import { ChatMessage, initialMessages } from '@/lib/mockData';

export default function AIChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    const text = input.trim();
    if (!text) return;

    setMessages((prev) => [
      ...prev,
      { role: 'user', text },
      {
        role: 'ai',
        text: "Got it! Let me check that for you. 🍽️ Type anything else or I'll confirm your full order shortly.",
      },
    ]);
    setInput('');
  };

  return (
    <div
      id="chat"
      className="flex flex-col bg-[#111] border border-white/5 rounded-2xl overflow-hidden h-[520px]"
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-white/5 shrink-0">
        <div className="w-8 h-8 rounded-full bg-pink-500/20 flex items-center justify-center text-pink-400 text-sm font-bold">
          AI
        </div>
        <div>
          <p className="text-white text-sm font-semibold">Kitchen Assistant</p>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full" />
            <span className="text-gray-500 text-xs">Online now</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[82%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-pink-500 text-white rounded-br-sm'
                  : 'bg-white/5 text-gray-200 rounded-bl-sm'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-white/5 flex gap-2 shrink-0">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="e.g. I want jollof rice and a drink..."
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-600 outline-none focus:border-pink-500/50 transition-colors"
        />
        <button
          onClick={sendMessage}
          className="px-4 py-2.5 bg-pink-500 hover:bg-pink-600 text-white rounded-xl text-sm font-semibold transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
