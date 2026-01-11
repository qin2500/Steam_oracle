import { GameResult } from '@/lib/types';
import { GameCard } from './GameCard';

interface GameGridProps {
    results: GameResult[];
    isLoading: boolean;
    onSelectGame: (game: GameResult) => void;
}

export function GameGrid({ results, isLoading, onSelectGame }: GameGridProps) {
    if (isLoading && results.length === 0) {
        // Skeletons could go here
        return null;
    }

    if (results.length === 0) return null;

    return (
        <div className="w-full max-w-7xl mx-auto animate-in">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.map((game) => (
                    <GameCard
                        key={game.id}
                        game={game}
                        onClick={() => onSelectGame(game)}
                    />
                ))}
            </div>
        </div>
    );
}
