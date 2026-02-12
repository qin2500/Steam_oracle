# Steam Oracle

Steam Oracle is a web application that provides insights and analysis on Steam games using a vector search engine and LLM integration.

> [!NOTE]
> This project was developed in collaboration with **Blue Ocean Games**.

## Project Structure

-   **Backend**: Python/FastAPI application with PostgreSQL and Vector Search.
-   **Frontend**: Next.js application for the user interface.

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   Node.js (for local frontend development outside Docker)
-   Python 3.10+ (for local backend development outside Docker)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Steam_oracle
    ```

2.  **Configure Environment Variables:**

    *   **Backend:** Copy the example environment file and fill in your details.
        ```bash
        cd Backend
        cp .env.example .env
        ```
        *   `OPENROUTER_API_KEY`: Your API key for OpenRouter.
        *   `DB_URI`: Connection string for your PostgreSQL database.

    *   **Frontend:** Copy the example environment file.
        ```bash
        cd ../Frontend
        cp .env.example .env
        ```
        *   `NEXT_PUBLIC_API_URL`: URL of the backend API (default: `http://localhost:8000`).

3.  **Run with Docker Compose:**
    ```bash
    # From the root directory
    docker-compose up --build
    ```

## Data Privacy & Database Schema

**Note:** The Steam database used in this project contains proprietary or restricted data that cannot be shared publicly. Therefore, the database dump is not included in this repository.

However, if you wish to set up your own database to run this application, you can use the schema below.

### Database Schema

```sql
TABLE: alembic_version
----------------------
  version_num               character varying    NULLABLE: NO

TABLE: games
------------
  game_id                   character varying    NULLABLE: NO
  timestamp                 timestamp            NULLABLE: YES
  name                      character varying    NULLABLE: YES
  release_date              timestamp            NULLABLE: YES
  is_free                   boolean              NULLABLE: YES
  price                     integer              NULLABLE: YES
  developer                 character varying    NULLABLE: YES
  publisher                 character varying    NULLABLE: YES
  positive_reviews          integer              NULLABLE: YES
  negative_reviews          integer              NULLABLE: YES
  genres                    text                 NULLABLE: YES
  reviews_processed         boolean              NULLABLE: YES
  reviews_all_langs_processed boolean            NULLABLE: YES
  reviews_processed_at      timestamp            NULLABLE: YES

TABLE: ownerships
-----------------
  ownership_id              character varying    NULLABLE: NO
  timestamp                 timestamp            NULLABLE: YES
  game_id                   character varying    NULLABLE: YES
  user_id                   character varying    NULLABLE: YES
  playtime_forever          integer              NULLABLE: YES
  last_played               timestamp            NULLABLE: YES

TABLE: reviews
--------------
  review_id                 character varying    NULLABLE: NO
  timestamp                 timestamp            NULLABLE: YES
  user_id                   character varying    NULLABLE: YES
  game_id                   character varying    NULLABLE: YES
  language                  character varying    NULLABLE: YES
  text                      text                 NULLABLE: YES
  time_created              timestamp            NULLABLE: YES
  time_updated              timestamp            NULLABLE: YES
  is_positive               boolean              NULLABLE: YES
  votes_up                  integer              NULLABLE: YES
  votes_funny               integer              NULLABLE: YES
  weighted_vote_score       double precision     NULLABLE: YES
  comment_count             integer              NULLABLE: YES
  steam_purchase            boolean              NULLABLE: YES
  received_for_free         boolean              NULLABLE: YES
  written_during_early_access boolean            NULLABLE: YES
  playtime_at_review        integer              NULLABLE: YES
  playtime_forever          integer              NULLABLE: YES
  playtime_last_two_weeks   integer              NULLABLE: YES
  last_played               timestamp            NULLABLE: YES

TABLE: users
------------
  user_id                   character varying    NULLABLE: NO
  timestamp                 timestamp            NULLABLE: YES
  personal_name             character varying    NULLABLE: YES
  real_name                 character varying    NULLABLE: YES
  profile_url               character varying    NULLABLE: YES
  time_created              timestamp            NULLABLE: YES
  country_code              character varying    NULLABLE: YES
  ownerships_processed      boolean              NULLABLE: YES
```