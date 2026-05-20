export default function Hero() {
  return (
    <section className="relative overflow-hidden py-24 px-6 text-center">
      {/* Background glow */}
      <div className="absolute inset-0 -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[700px] h-[400px] bg-pink-500/10 rounded-full blur-3xl" />
      </div>

      <div className="max-w-3xl mx-auto">
        {/* Badge */}
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-pink-500/10 border border-pink-500/20 text-pink-400 text-xs font-medium mb-6">
          <span className="w-1.5 h-1.5 bg-pink-400 rounded-full animate-pulse" />
          Open Now &middot; 24/7 AI Ordering
        </span>

        <h1 className="text-5xl md:text-6xl font-extrabold text-white tracking-tight mb-5 leading-tight">
          Mama&apos;s Kitchen
        </h1>

        <p className="text-lg text-gray-400 max-w-xl mx-auto mb-10 leading-relaxed">
          Fresh homemade Nigerian food, ordered through an AI assistant.
          Browse the menu, chat your order — done in seconds.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <a
            href="#menu"
            className="px-7 py-3 bg-pink-500 hover:bg-pink-600 text-white font-semibold rounded-xl transition-colors text-sm"
          >
            Browse Menu
          </a>
          <a
            href="#chat"
            className="px-7 py-3 bg-white/5 hover:bg-white/10 border border-white/10 text-white font-semibold rounded-xl transition-colors text-sm"
          >
            Chat &amp; Order
          </a>
        </div>

        {/* Stats row */}
        <div className="mt-16 flex flex-wrap justify-center gap-8 text-center">
          {[
            { value: '6', label: 'Menu Items' },
            { value: '< 2 min', label: 'Order Time' },
            { value: '24/7', label: 'Always Open' },
          ].map(({ value, label }) => (
            <div key={label}>
              <p className="text-white font-bold text-2xl">{value}</p>
              <p className="text-gray-600 text-xs mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
