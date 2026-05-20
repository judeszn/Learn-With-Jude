import { Order } from '@/lib/mockData';

const statusConfig = {
  standby: {
    label: 'Standby',
    color: 'text-yellow-400',
    bg: 'bg-yellow-400/10 border-yellow-400/20',
    dot: 'bg-yellow-400',
    message: 'Waiting for the kitchen to confirm your order.',
  },
  accepted: {
    label: 'Accepted',
    color: 'text-green-400',
    bg: 'bg-green-400/10 border-green-400/20',
    dot: 'bg-green-400',
    message: 'Kitchen accepted! Estimated time: 25–35 min.',
  },
  preparing: {
    label: 'Preparing',
    color: 'text-blue-400',
    bg: 'bg-blue-400/10 border-blue-400/20',
    dot: 'bg-blue-400',
    message: "Your food is being prepared. Won't be long!",
  },
  ready: {
    label: 'Ready',
    color: 'text-pink-400',
    bg: 'bg-pink-400/10 border-pink-400/20',
    dot: 'bg-pink-400',
    message: 'Your order is ready for pickup or delivery!',
  },
};

type Props = {
  order: Order;
};

export default function OrderCard({ order }: Props) {
  const config = statusConfig[order.status];

  return (
    <div className="bg-[#111] border border-white/5 rounded-2xl p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-600 text-[10px] uppercase tracking-widest mb-0.5">Order ID</p>
          <p className="text-white font-mono font-bold text-sm">{order.id}</p>
        </div>
        <span
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${config.bg} ${config.color}`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${config.dot} animate-pulse`} />
          {config.label}
        </span>
      </div>

      {/* Status message */}
      <p className="text-gray-500 text-xs">{config.message}</p>

      {/* Items list */}
      <div className="space-y-2 pt-1">
        {order.items.map(({ item, qty }, i) => (
          <div key={i} className="flex justify-between items-center">
            <span className="text-gray-300 text-sm">
              {item.emoji} {item.name}{' '}
              <span className="text-gray-600 text-xs">×{qty}</span>
            </span>
            <span className="text-gray-400 text-sm">
              ₦{(item.price * qty).toLocaleString()}
            </span>
          </div>
        ))}
      </div>

      {/* Total */}
      <div className="border-t border-white/5 pt-3 flex justify-between items-center">
        <span className="text-gray-500 text-sm">Total</span>
        <span className="text-white font-bold">₦{order.total.toLocaleString()}</span>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>{order.customerName}</span>
        <span>{order.createdAt}</span>
      </div>
    </div>
  );
}
