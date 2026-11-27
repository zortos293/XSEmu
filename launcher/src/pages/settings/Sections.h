#pragma once

#include <functional>

class QWidget;
class Services;

// One builder per Settings section (Settings pages). Each returns the section's page; the page keeps
// itself up to date from Services. Backing of every control: see the README's Settings table.
namespace SettingsSections {

QWidget* console(Services& s);
QWidget* network(Services& s);
QWidget* shaders(Services& s);
QWidget* runProfile(Services& s);
QWidget* overlay(Services& s);
QWidget* gameFolders(Services& s, std::function<void()> addFolder);
QWidget* input(Services& s);
QWidget* graphics(Services& s);
QWidget* logs(Services& s);
QWidget* updates(Services& s);
QWidget* about(Services& s);

}
