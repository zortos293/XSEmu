#pragma once

#include "core/Game.h"
#include "widgets/Card.h"

#include <array>

class Button;
class Label;
class QLabel;
class Services;

// The Artwork card on Game details: the SteamGridDB game the art comes from (with Change... to pick
// another, or go back to the automatic match), then one row per kind (cover, banner, logo, icon) with
// a preview, where it came from and its artist, and Choose image... / Reset for the user's own file.
// Refresh artwork in the header fetches the art again.
class ArtworkCard : public Card {
    Q_OBJECT
public:
    explicit ArtworkCard(Services& services, QWidget* parent = nullptr);
    void setGame(const Game& game);
    void chooseMatch(); // the "Choose the SteamGridDB game" dialog

private:
    struct Slot {
        QLabel* preview = nullptr;
        Label* source = nullptr;
        Button* reset = nullptr;
    };

    void refresh();

    Services& m_s;
    Game m_game;
    Label* m_match;
    Button* m_change;
    Button* m_refresh;
    std::array<Slot, 4> m_slots;
};
