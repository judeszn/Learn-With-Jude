import { MenuItem } from '@/lib/mockData';

type Props = {
  item: MenuItem;
  onOrder: (item: MenuItem) => void;
};

export default function FoodCard({ item, onOrder }: Props) {
  return (
    <div className="relative bg-[#111] border border-white/5 rounded-2xl p-5 hover:border-pink-500/25 transition-all duration-200 flex flex-col">
      {item.popular && (
        <span className="absolute top-3 right-3 px-2 py-0.5 bg-pink-500/15 border border-pink-500/20 text-pink-400 text-[10px] font-bold rounded-full uppercase tracking-widest">
          Popular
        </span>
      )}

      <div className="text-4xl mb-3">{item.emoji}</div>
      <span className="text-[10px] font-semibold text-pink-400 uppercase tracking-widest mb-1">
        {item.category}
      </span>
      <h3 className="text-white font-semibold text-sm mb-1">{item.name}</h3>
      <p className="text-gray-500 text-xs leading-relaxed mb-4 flex-1">
        {item.description}
      </p>

      <div className="flex items-center justify-between mt-auto">
        <span className="text-white font-bold text-sm">
          ₦{item.price.toLocaleString()}
        </span>
        <button
          onClick={() => onOrder(item)}
          className="text-xs px-3 py-1.5 bg-pink-500 hover:bg-pink-600 text-white rounded-lg font-semibold transition-colors"
        >
          + Add
        </button>
      </div>
    </div>
  );
}
