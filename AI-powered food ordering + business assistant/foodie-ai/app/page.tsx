'use client';

import { useState } from 'react';
import Hero from '@/components/Hero';
import MenuSection from '@/components/MenuSection';
import AIChatPanel from '@/components/AIChatPanel';
import OrderCard from '@/components/OrderCard';
import { MenuItem, mockOrder, Order } from '@/lib/mockData';

export default function Home() {
  const [order] = useState<Order>(mockOrder);

  const handleAddItem = (item: MenuItem) => {
    const el = document.getElementById('chat');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
    console.log('Added to order:', item.name);
  };

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-white/5 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">🍽️</span>
            <span className="text-white font-bold text-sm tracking-tight">
              Mama&apos;s Kitchen
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <a href="#menu" className="hover:text-white transition-colors">Menu</a>
            <a href="#chat" className="hover:text-white transition-colors">Order</a>
            <span className="flex items-center gap-1.5 text-green-400">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full" />
              Open
            </span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <Hero />

      <div className="max-w-6xl mx-auto px-6"><div className="border-t border-white/5" /></div>

      {/* Menu */}
      <MenuSection onOrder={handleAddItem} />

      <div className="max-w-6xl mx-auto px-6"><div className="border-t border-white/5" /></div>

      {/* Chat + Order */}
      <section className="py-16 px-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-1">Place Your Order</h2>
          <p className="text-gray-500 text-sm">
            Chat with the AI assistant or check your current order status.
          </p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AIChatPanel />
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-white font-semibold text-sm">Current Order</h3>
              <span className="text-gray-600 text-xs">Live status</span>
            </div>
            <OrderCard order={order} />
            <div className="bg-white/3 border border-white/5 rounded-xl p-4 space-y-2">
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest">
                How it works
              </p>
              <ol className="text-gray-500 text-xs space-y-1.5 list-decimal list-inside leading-relaxed">
                <li>Chat your order with the AI assistant</li>
                <li>Order goes into <span className="text-yellow-400">Standby</span></li>
                <li>Kitchen confirms → status updates</li>
                <li>Pay on delivery or pickup</li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6 text-center text-gray-600 text-xs">
        <p>Mama&apos;s Kitchen &copy; 2026 &mdash; Built with AI on Learn-With-Jude</p>
      </footer>
    </main>
  );
}


