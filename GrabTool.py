import pyautogui

print("Déplacez votre souris sur l'élément cible, puis appuyez sur Ctrl+C pour arrêter.")
try:
    while True:
        x, y = pyautogui.position()
        print(f"Position de la souris : X={x}, Y={y}", end="\r")
except KeyboardInterrupt:
    print("\nTerminé.")
