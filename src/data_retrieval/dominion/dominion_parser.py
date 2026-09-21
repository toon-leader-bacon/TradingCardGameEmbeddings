#!/usr/bin/env python3

import re
import sys
import time

import selenium.webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import selenium.webdriver.support.expected_conditions as EC


def create_driver():

    opts = selenium.webdriver.ChromeOptions()
    opts.add_argument("--incognito")
    return selenium.webdriver.Chrome(options=opts)


def login(driver, username):

    driver.get("https://dominion.games")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username-input"))
    ).send_keys(username)
    # pass_elem = driver.find_element(By.NAME, 'password')
    # pass_elem.send_keys(getpass.getpass(f'{username} password: '))
    # pass_elem.send_keys(Keys.RETURN)


def wait_end(driver):

    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[normalize-space()="The game has ended."]')
        )
    )


def close_game(driver):

    driver.find_element(By.XPATH, '//button[@ng-click="$ctrl.ok()"]').click()
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="Leave Table"]')
        )
    ).click()


def resign_game(driver):

    leave_game(driver)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[@ng-click="$ctrl.grant()"]')
        )
    ).click()
    wait_end(driver)
    close_game(driver)


def leave_game(driver):

    driver.find_element(By.XPATH, '//div[@ng-click="$ctrl.resignOrLeave()"]').click()


def get_log(driver):

    raw = driver.find_element(By.XPATH, '//*[@class="game-log"]').get_attribute(
        "innerHTML"
    )

    raw = raw.replace("\n", "")
    matches = re.findall(
        r'(class="[^"]*"|style="padding-left: [^%]*|<span>[^<]*</span>)', raw
    )

    indents = ["0"]
    indent = "0"
    log = ""

    for match in matches:
        if match.startswith("class"):
            if "new-turn-line" in match:
                log += "\n\n"
        elif match.startswith("style"):
            indent = match.split()[-1]
            if indent not in indents:
                indents.append(indent)
            log += "\n"
            log += " " * 4 * indents.index(indent)
        else:
            log += match[6:-7]

    return log


PROVINCE_PILE = '//div[contains(@style, "mini-province.webp")]/..'


def wait_for_decision(driver):
    """Block until the replay has loaded at least one decision."""
    decision = driver.find_element(By.ID, "table-replay-decision")
    while True:
        try:
            if int(driver.execute_script("return arguments[0].value", decision)) > 0:
                return
        except Exception:
            pass
        time.sleep(1)


def game_ended(driver):
    try:
        wait_end(driver)
    except Exception:
        return False
    return True


def read_province_status(driver):
    """Return (provinces_gone, maybe_provinces).

    maybe_provinces is None when the provinces are already gone; otherwise it
    says whether the last province could be bought with the coins left.
    """
    try:
        driver.find_element(By.XPATH, f'{PROVINCE_PILE}/div[@class="empty-card-cross"]')
    except Exception:
        pass
    else:
        return True, None

    provinces_left = int(
        driver.find_element(
            By.XPATH, f'{PROVINCE_PILE}/div[contains(@class, "counter-layer")]/*[1]'
        ).text
    )
    province_cost = int(
        driver.find_element(
            By.XPATH,
            f'{PROVINCE_PILE}/div[contains(@class, "coins-layer")]/*[1]/*[1]',
        ).text
    )
    coins_left = int(
        driver.find_element(
            By.XPATH, '//counter-element[contains(@style, "coin.png")]/*[1]'
        ).text
    )
    return False, provinces_left == 1 and coins_left >= province_cost


def start_table(driver1, driver2, game_id, name1, swap):
    """Load the old game on driver1 and seat driver2 as the opponent."""
    WebDriverWait(driver1, 10 if swap else 999).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="New Table"]')
        )
    ).click()
    WebDriverWait(driver1, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[normalize-space()="Load Old Game"]')
        )
    ).click()
    WebDriverWait(driver1, 10).until(
        EC.presence_of_element_located((By.ID, "table-replay-id"))
    ).send_keys(game_id)
    driver1.find_element(By.XPATH, '//div[normalize-space()="Load from End"]').click()
    wait_for_decision(driver1)

    WebDriverWait(driver1, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@class="table-add-bot-icon"]'))
    ).click()

    if swap:
        WebDriverWait(driver1, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@class="table-order-down"]'))
        ).click()
        time.sleep(1)

    friend_join_button = (
        f'//td[@class="friend-activities-name-column" and .="{name1}"]/../*[2]/button'
    )
    WebDriverWait(driver2, 10 if swap else 999).until(
        EC.presence_of_element_located(
            (By.XPATH, '//button[normalize-space()="Automatch"]')
        )
    ).click()
    WebDriverWait(driver2, 10).until(
        EC.presence_of_element_located((By.XPATH, friend_join_button))
    ).click()
    WebDriverWait(driver2, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[normalize-space()="Ready"]'))
    )

    driver1.find_element(By.XPATH, '//div[normalize-space()="Ready"]').click()


def load_game(driver1, driver2, game_id, name1, swap):
    start_table(driver1, driver2, game_id, name1, swap)

    if game_ended(driver1):
        wait_end(driver2)
        close_game(driver2)
        close_game(driver1)
        return load_game(driver1, driver2, game_id, name1, True)

    log = get_log(driver2)

    if log.split("\n")[-1].startswith("Turn "):
        leave_game(driver2)
        resign_game(driver1)
        return load_game(driver1, driver2, game_id, name1, True)

    provinces, maybe_provinces = read_province_status(driver1)

    leave_game(driver2)
    resign_game(driver1)
    return log, provinces, maybe_provinces


driver1 = create_driver()
driver2 = create_driver()

login(driver1, sys.argv[1])
login(driver2, sys.argv[2])

for game_id in sys.argv[3:]:
    with open(f"{game_id}.log", "w") as f:
        log, complete, provinces = load_game(
            driver1, driver2, game_id, sys.argv[1], False
        )
        print(log, file=f)
        print(complete, provinces, file=f)
