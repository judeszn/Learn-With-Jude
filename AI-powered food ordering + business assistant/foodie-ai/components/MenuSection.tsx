'use client';

import { useState } from 'react';
import { menuItems, MenuItem } from '@/lib/mockData';
import FoodCard from './FoodCard';

const categories = ['All', ...Array.from(new Set(menuItems.map((i) => i.category)))];

type Props = {
  onOrder: (item: MenuItem) => void;
};

export default function MenuSection({ onOrder }: Props) {
  const [active, setActive] = useState('All');

  const filtered =
    active === 'All' ? menuItems : menuItems.filter((i) => i.category === active);

  return (
    <section id="menu" className="py-16 px-6 max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-1">Our Menu</h2>
        <p className="text-gray-500 text-sm">
          Fresh made daily. Order via AI chat below or tap &quot;+ Add&quot; on any item.
        </p>
      </div>

      {/* Category filter */}
      <div className="flex gap-2 flex-wrap mb-8">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setActive(cat)}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-colors border ${
              active === cat
                ? 'bg-pink-500 border-pink-500 text-white'
                : 'bg-transparent border-white/10 text-gray-400 hover:border-white/20 hover:text-gray-300'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((item) => (
          <FoodCard key={item.id} item={item} onOrder={onOrder} />
        ))}
      </div>
    </section>
  );
}
