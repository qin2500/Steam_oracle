import { GameResult } from '@/lib/types';
import { ThumbsUp, ThumbsDown, Info } from 'lucide-react';

interface GameCardProps {
    game: GameResult;
    onClick: () => void;
}

export function GameCard({ game, onClick }: GameCardProps) {
    // Calculate color based on score
    const getScoreColor = (score: number) => {
        if (score >= 90) return '#66C0F4'; // Steam Blue
        if (score >= 70) return '#A3D164'; // Positive Green
        return '#B9A074'; // Mixed/Average
    };

    const scoreColor = getScoreColor(game.score);

    return (
        <div
            onClick={onClick}
            className="steam-card p-4 flex flex-col h-full hover:bg-steam-blue/20 group cursor-pointer"
        >
            {/* Header / Top Section */}
            <div className="flex justify-between items-start mb-3">
                <h3 className="text-xl font-bold text-white group-hover:text-steam-light-blue transition-colors">
                    {game.name}
                </h3>
                <div
                    className="w-3 h-3 rounded-full shadow-[0_0_8px_currentColor] transition-colors duration-300 shrink-0 mt-2"
                    style={{ backgroundColor: scoreColor, color: scoreColor }}
                    title={`Match Confidence: ${game.score}%`}
                />
            </div>

            {/* Image Placeholder */}
            {/* Image Placeholder */}
            <div className="w-full h-32 mb-4 rounded overflow-hidden">
                <img
                    src="/place_holder.jpg"
                    alt="Game Cover"
                    className="w-full h-full object-cover"
                />
            </div>

            {/* Summary */}
            <p className="text-sm text-steam-text/80 mb-4 line-clamp-3">
                {game.summary}
            </p>

            {/* Reasoning Highlight */}
            <div className="mb-4 bg-black/20 p-3 rounded border-l-2 border-steam-light-blue">
                <p className="text-xs text-steam-light-blue italic">"{game.reasoning}"</p>
            </div>

        </div>
    );
}
