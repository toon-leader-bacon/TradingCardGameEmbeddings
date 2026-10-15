#!/usr/bin/env python3
"""Two-browser Selenium scraper that rebuilds a dominion.games game log.

Prototype from the Dominion Discord; see todo.md, "The live path", for
how it works and for the scraping pause the site owners asked for. Do
not run it until that pause is lifted.

driver1 loads the old game from its end and adds a bot; driver2 joins
as the opponent and reads the log from its side. A game that ends on
load, or whose log ends on a "Turn" line, is retried with the seats
swapped.

Selenium is not in requirements.txt; install it by hand to run this.

Usage:
    python -m src.data_retrieval.dominion.dominion_parser \\
        <username1> <username2> <game_id> [<game_id> ...]

Writes <game_id>.log in the working directory for each game: the log
text, then a line "<provinces_gone> <last_province_affordable>".
"""

import sys
import time
from typing import NamedTuple

import selenium.webdriver
import selenium.webdriver.support.expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.wait import WebDriverWait

from src.data_retrieval.dominion.game_log_text import game_log_text

_SITE_URL = "https://dominion.games"
_PROVINCE_PILE = '//div[contains(@style, "mini-province.webp")]/..'
_GAME_ENDED_LINE = '//div[normalize-space()="The game has ended."]'
_DECISION_POLL_SECONDS = 1
# How long the first seat waits for the lobby: a human may still be
# finishing the login by hand.
_FIRST_LOBBY_WAIT_SECONDS = 999
_STEP_WAIT_SECONDS = 10
_GAME_END_WAIT_SECONDS = 5


class GameResult(NamedTuple):
    """One scraped game: its log and the Province pile at the end."""

    log: str
    provinces_gone: bool
    # None when the Provinces are gone; otherwise whether the last one
    # was affordable with the coins left.
    last_province_affordable: bool | None


def create_driver() -> WebDriver:
    """Open an incognito Chrome session.

    Inputs: none. Output: the new WebDriver.
    Side effects: launches a Chrome process.
    Exceptions: whatever Selenium raises when Chrome can't start.
    """
    options = selenium.webdriver.ChromeOptions()
    options.add_argument("--incognito")
    return selenium.webdriver.Chrome(options=options)


def login(driver: WebDriver, username: str) -> None:
    """Open the site and type username into the login form.

    The password is left to a human (the prototype's password lines
    were commented out), so the caller waits on the lobby afterwards.

    Inputs: driver; username. Output: none.
    Side effects: navigates driver and types into the page.
    Exceptions: TimeoutException if the login form never appears.
    """
    driver.get(_SITE_URL)
    WebDriverWait(driver, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.ID, "username-input"))
    ).send_keys(username)


def wait_end(driver: WebDriver) -> None:
    """Wait for the "The game has ended." line.

    Inputs: driver. Output: none. Side effects: none.
    Exceptions: TimeoutException if it does not appear in time.
    """
    WebDriverWait(driver, _GAME_END_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.XPATH, _GAME_ENDED_LINE))
    )


def close_game(driver: WebDriver) -> None:
    """Dismiss the end-of-game dialog and leave the table.

    Inputs: driver. Output: none. Side effects: clicks in the page.
    Exceptions: NoSuchElementException / TimeoutException if the
        dialog or the Leave Table button is missing.
    """
    driver.find_element(By.XPATH, '//button[@ng-click="$ctrl.ok()"]').click()
    WebDriverWait(driver, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="Leave Table"]')
        )
    ).click()


def leave_game(driver: WebDriver) -> None:
    """Click the resign-or-leave control.

    Inputs: driver. Output: none. Side effects: clicks in the page.
    Exceptions: NoSuchElementException if the control is missing.
    """
    driver.find_element(By.XPATH, '//div[@ng-click="$ctrl.resignOrLeave()"]').click()


def resign_game(driver: WebDriver) -> None:
    """Resign, wait for the game to end, then leave the table.

    Inputs: driver. Output: none. Side effects: clicks in the page.
    Exceptions: as leave_game, wait_end and close_game.
    """
    leave_game(driver)
    WebDriverWait(driver, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[@ng-click="$ctrl.grant()"]')
        )
    ).click()
    wait_end(driver)
    close_game(driver)


def read_log(driver: WebDriver) -> str:
    """Read the game log on driver's page as plain text.

    Inputs: driver. Output: the log text (see game_log_text).
    Side effects: none.
    Exceptions: NoSuchElementException if there is no game log.
    """
    log_html = driver.find_element(By.XPATH, '//*[@class="game-log"]').get_attribute(
        "innerHTML"
    )
    return game_log_text(log_html or "")


def wait_for_decision(driver: WebDriver) -> None:
    """Block until the replay has loaded at least one decision.

    Polls once a second with no time limit, as the prototype did.

    Inputs: driver. Output: none. Side effects: none.
    Exceptions: NoSuchElementException if the replay control is missing.
    """
    decision = driver.find_element(By.ID, "table-replay-decision")
    while True:
        value = driver.execute_script("return arguments[0].value", decision)
        try:
            if int(value) > 0:
                return
        except (TypeError, ValueError):
            pass  # not loaded yet: the field is empty
        time.sleep(_DECISION_POLL_SECONDS)


def has_game_ended(driver: WebDriver) -> bool:
    """Whether the game-ended line appears within wait_end's timeout.

    Inputs: driver. Output: bool. Side effects: none. Exceptions: none.
    """
    try:
        wait_end(driver)
    except TimeoutException:
        return False
    return True


def read_province_status(driver: WebDriver) -> tuple[bool, bool | None]:
    """Read the Province pile at the end of the game.

    Inputs: driver.
    Output: (provinces_gone, last_province_affordable); the second is
        None when the Provinces are gone, else whether exactly one is
        left and the coins in play cover its cost.
    Side effects: none.
    Exceptions: NoSuchElementException / ValueError if the pile's
        counters are missing or not numbers.
    """
    try:
        driver.find_element(
            By.XPATH, f'{_PROVINCE_PILE}/div[@class="empty-card-cross"]'
        )
    except NoSuchElementException:
        pass  # the pile is not empty
    else:
        return True, None

    provinces_left = int(
        driver.find_element(
            By.XPATH, f'{_PROVINCE_PILE}/div[contains(@class, "counter-layer")]/*[1]'
        ).text
    )
    province_cost = int(
        driver.find_element(
            By.XPATH,
            f'{_PROVINCE_PILE}/div[contains(@class, "coins-layer")]/*[1]/*[1]',
        ).text
    )
    coins_left = int(
        driver.find_element(
            By.XPATH, '//counter-element[contains(@style, "coin.png")]/*[1]'
        ).text
    )
    return False, provinces_left == 1 and coins_left >= province_cost


def start_table(
    driver1: WebDriver, driver2: WebDriver, game_id: str, name1: str, swap: bool
) -> None:
    """Load the old game on driver1 and seat driver2 as the opponent.

    Inputs:
        driver1, driver2: the two logged-in sessions.
        game_id: the dominion.games game to load.
        name1: driver1's username, which driver2 joins through.
        swap: move driver1's seat down (a retry). The first attempt also
            waits up to _FIRST_LOBBY_WAIT_SECONDS for each lobby, to
            leave time for a manual login.
    Output: none.
    Side effects: creates a table and readies both players.
    Exceptions: TimeoutException / NoSuchElementException if the page
        does not behave as expected.
    """
    lobby_wait = _STEP_WAIT_SECONDS if swap else _FIRST_LOBBY_WAIT_SECONDS
    WebDriverWait(driver1, lobby_wait).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="New Table"]')
        )
    ).click()
    WebDriverWait(driver1, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[normalize-space()="Load Old Game"]')
        )
    ).click()
    WebDriverWait(driver1, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.ID, "table-replay-id"))
    ).send_keys(game_id)
    driver1.find_element(By.XPATH, '//div[normalize-space()="Load from End"]').click()
    wait_for_decision(driver1)

    WebDriverWait(driver1, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.XPATH, '//*[@class="table-add-bot-icon"]'))
    ).click()

    if swap:
        WebDriverWait(driver1, _STEP_WAIT_SECONDS).until(
            EC.presence_of_element_located((By.XPATH, '//*[@class="table-order-down"]'))
        ).click()
        time.sleep(1)

    friend_join_button = (
        f'//td[@class="friend-activities-name-column" and .="{name1}"]/../*[2]/button'
    )
    WebDriverWait(driver2, lobby_wait).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="Automatch"]')
        )
    ).click()
    WebDriverWait(driver2, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.XPATH, friend_join_button))
    ).click()
    WebDriverWait(driver2, _STEP_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.XPATH, '//div[normalize-space()="Ready"]'))
    )

    driver1.find_element(By.XPATH, '//div[normalize-space()="Ready"]').click()


def load_game(
    driver1: WebDriver, driver2: WebDriver, game_id: str, name1: str
) -> GameResult:
    """Replay game_id and read its log and Province pile.

    The first attempt keeps the original seats. If the game ends as soon
    as it loads, or the log ends on a "Turn" line, both tables are
    closed and it is retried with the seats swapped, with no limit on
    retries (as the prototype did).

    Inputs: as start_table.
    Output: the GameResult.
    Side effects: drives both sessions; leaves both back in the lobby.
    Exceptions: as start_table.
    """
    swap = False
    while True:
        start_table(driver1, driver2, game_id, name1, swap)
        swap = True

        if has_game_ended(driver1):
            wait_end(driver2)
            close_game(driver2)
            close_game(driver1)
            continue

        log = read_log(driver2)
        if log.split("\n")[-1].startswith("Turn "):
            leave_game(driver2)
            resign_game(driver1)
            continue

        provinces_gone, last_province_affordable = read_province_status(driver1)
        leave_game(driver2)
        resign_game(driver1)
        return GameResult(log, provinces_gone, last_province_affordable)


def main() -> None:
    """Scrape every game id on the command line; see the module docstring.

    Inputs: sys.argv: username1, username2, then one or more game ids.
    Output: none.
    Side effects: launches two Chrome sessions; writes <game_id>.log in
        the working directory per game.
    Exceptions: ValueError if fewer than two usernames are given; as
        load_game.
    """
    username1, username2, *game_ids = sys.argv[1:]
    driver1 = create_driver()
    driver2 = create_driver()

    login(driver1, username1)
    login(driver2, username2)

    for game_id in game_ids:
        result = load_game(driver1, driver2, game_id, username1)
        with open(f"{game_id}.log", "w") as log_file:
            print(result.log, file=log_file)
            print(result.provinces_gone, result.last_province_affordable, file=log_file)


if __name__ == "__main__":
    main()
