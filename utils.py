import os
import pandas as pd

def list_teams():
    """
    stats klasöründeki tüm takımların listesini döndürür.
    """
    teams = []
    for file in os.listdir('stats'):
        if file.endswith('.csv'):
            teams.append(file[:-4])
    return sorted(teams)

def display_teams():
    """
    Tüm takımları numaralandırılmış şekilde gösterir.
    """
    teams = list_teams()
    print("\nMevcut Takımlar:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    return teams

def get_team_selection():
    """
    Kullanıcıdan iki takım seçmesini ister.
    """
    teams = display_teams()
    while True:
        try:
            print("\nEv sahibi takımın numarasını girin:")
            home_idx = int(input()) - 1
            print("Deplasman takımının numarasını girin:")
            away_idx = int(input()) - 1
            
            if 0 <= home_idx < len(teams) and 0 <= away_idx < len(teams):
                return teams[home_idx], teams[away_idx]
            else:
                print("Geçersiz takım numarası! Lütfen tekrar deneyin.")
        except ValueError:
            print("Lütfen geçerli bir sayı girin!") 