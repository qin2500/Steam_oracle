import { GameResult } from '@/lib/types';
import { X } from 'lucide-react';
import { useEffect } from 'react';

interface GameModalProps {
    game: GameResult;
    onClose: () => void;
}

export function GameModal({ game, onClose }: GameModalProps) {
    // Prevent scrolling when modal is open
    useEffect(() => {
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, []);

    // Close on Escape key
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [onClose]);

    const getDotColor = (status: string) => {
        // Logic for criteria colored dots
        if (status === 'met') return 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]';
        if (status === 'missing' || status === 'violated') return 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]';
        return 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]';
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal Content */}
            <div className="relative w-full max-w-2xl bg-[#1b2838] border border-steam-blue shadow-2xl rounded-lg overflow-hidden animate-in flex flex-col max-h-[90vh]">

                {/* Header Image */}
                <div className="relative h-48 sm:h-64 shrink-0">
                    <img
                        src="/place_holder.jpg"
                        alt={game.name}
                        className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-[#1b2838] to-transparent opacity-90" />

                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-2 bg-black/40 hover:bg-black/60 rounded-full text-white/80 hover:text-white transition-colors"
                    >
                        <X size={20} />
                    </button>

                    <div className="absolute bottom-4 left-6">
                        <h2 className="text-3xl font-bold text-white shadow-black drop-shadow-lg">{game.name}</h2>
                    </div>
                </div>

                {/* Scrollable Content */}
                <div className="p-6 overflow-y-auto custom-scrollbar">

                    {/* Reasoning & Summary */}
                    <div className="mb-6 space-y-4">
                        <div className="bg-black/20 p-4 rounded border-l-4 border-steam-light-blue">
                            <p className="text-steam-light-blue italic text-lg">"{game.reasoning}"</p>
                        </div>
                        <p className="text-steam-text leading-relaxed">
                            {game.summary}
                        </p>
                    </div>

                    {/* Criteria Breakdown */}
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold text-white border-b border-white/10 pb-2">Match Breakdown</h3>
                        <div className="grid gap-3">
                            {game.assessments.map((asm, idx) => (
                                <div key={idx} className="flex items-start gap-4 p-3 rounded bg-black/20 hover:bg-black/30 transition-colors">
                                    {/* Indicator Dot */}
                                    <div className={`mt-1.5 w-3 h-3 rounded-full shrink-0 ${getDotColor(asm.status)}`} />

                                    <div className="flex-1">
                                        <div className="flex justify-between items-baseline mb-1">
                                            <span className="font-semibold text-steam-text">{asm.criterion || "General Fit"}</span>
                                        </div>
                                        <p className="text-sm text-steam-text/70">{asm.evidence}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
