
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
# =========================
# 1. CAU HINH
# =========================
FILE_CSV = "data/abc-11.csv"       # Doi thanh ten file cua ban, vi du abc-15.csv
START_DATE = "2025-01-01"    # Doi thanh ngay bat dau cua nam du lieu
OUTPUT_VIDEO = "stock_animation.mp4"
FPS = 20
DPI = 120

PRICE_COLUMNS = ["Open", "P1", "P2", "P3", "P4", "P5", "Close"]

# Mau theo quy uoc de bai
COLOR_REFERENCE = "#FFD700"  # Vang
COLOR_FLOOR = "#00A6FF"       # Xanh duong
COLOR_CEILING = "#B000FF"     # Tim
COLOR_DOWN = "#FF3030"        # Do
COLOR_UP = "#00D000"          # Xanh la
COLOR_EQUAL = COLOR_REFERENCE

# =========================
# 2. DOC VA LAM SACH DU LIEU
# =========================
def load_stock_data(file_path):
    raw = pd.read_csv(file_path)

    # Neu ten cot la cac gia tri so, file co the khong co header.
    # Doc lai voi header=None de khong mat dong dau tien.
    try:
        [float(str(c).replace(",", "")) for c in raw.columns]
        raw = pd.read_csv(file_path, header=None)
    except (ValueError, TypeError):
        pass

    if raw.shape[1] != 7:
        raise ValueError(
            f"File phai co dung 7 cot gia. Hien tai co {raw.shape[1]} cot."
        )

    raw.columns = PRICE_COLUMNS

    # Chuyen cac cot gia ve so; ho tro du lieu co dau phay ngan cach hang nghin.
    for col in PRICE_COLUMNS:
        raw[col] = (
            raw[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=PRICE_COLUMNS).reset_index(drop=True)

    if len(raw) == 0:
        raise ValueError("Khong co dong du lieu hop le trong file CSV.")

    # De bai co 365 ngay. Neu file co so dong khac 365, van tao ngay tuong ung.
    dates = pd.date_range(start=START_DATE, periods=len(raw), freq="D")
    raw.insert(0, "Date", dates)
    raw = raw.set_index("Date")

    return raw


df = load_stock_data(FILE_CSV)

# Them thong tin thoi gian
df["Month"] = df.index.month
df["Quarter"] = df.index.quarter

# Gia tham chieu cua ngay hien tai = gia Close ngay truoc
df["Ref"] = df["Close"].shift(1)
df["Floor"] = df["Ref"] * 0.90
df["Ceiling"] = df["Ref"] * 1.10

# Bien do trong ngay = gia cao nhat - gia thap nhat trong 7 moc gia
df["Daily_Range"] = df[PRICE_COLUMNS].max(axis=1) - df[PRICE_COLUMNS].min(axis=1)


# =========================
# 3. CAU 1 - THONG KE THEO THANG
# =========================
# Mac dinh dung gia dong cua (Close) de thong ke.
monthly = df.groupby("Month")["Close"].agg(
    Mean="mean",
    Std="std",
    Min="min",
    Max="max"
)
monthly["Range"] = monthly["Max"] - monthly["Min"]

print("\n========== CAU 1: THONG KE THEO THANG ==========")
print(monthly.round(2).to_string())

month_high_mean = monthly["Mean"].idxmax()
month_low_mean = monthly["Mean"].idxmin()
month_high_std = monthly["Std"].idxmax()
month_high_range = monthly["Range"].idxmax()

print(f"\nThang co gia Close trung binh cao nhat: thang {month_high_mean}")
print(f"Thang co gia Close trung binh thap nhat: thang {month_low_mean}")
print(f"Thang co do lech chuan lon nhat: thang {month_high_std}")
print(f"Thang co bien do Close (max-min) lon nhat: thang {month_high_range}")

# Bieu do cot gia trung binh
plt.figure(figsize=(10, 5))
plt.bar(monthly.index, monthly["Mean"], color="steelblue", edgecolor="black")
plt.xticks(range(1, 13))
plt.xlabel("Thang")
plt.ylabel("Gia dong cua trung binh (VND)")
plt.title("Gia dong cua trung binh theo thang")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("cau1_gia_trung_binh_theo_thang.png", dpi=200)
plt.show()

# Boxplot gia dong cua theo thang
monthly_box_data = [
    df.loc[df["Month"] == month, "Close"].dropna().values
    for month in range(1, 13)
]
plt.figure(figsize=(12, 6))
plt.boxplot(
    monthly_box_data,
    tick_labels=[str(m) for m in range(1, 13)],
    showfliers=True,
    patch_artist=True
)
plt.xlabel("Thang")
plt.ylabel("Gia dong cua (VND)")
plt.title("Boxplot gia dong cua theo thang")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("cau1_boxplot_theo_thang.png", dpi=200)
plt.show()


# =========================
# 4. CAU 2 - THONG KE THEO QUY
# =========================
quarterly = df.groupby("Quarter")["Close"].agg(
    Mean="mean",
    Std="std",
    Min="min",
    Max="max"
)
quarterly["Range"] = quarterly["Max"] - quarterly["Min"]

print("\n========== CAU 2: THONG KE THEO QUY ==========")
print(quarterly.round(2).to_string())

print(f"\nQuy on dinh nhat (Std nho nhat): Q{quarterly['Std'].idxmin()}")
print(f"Quy bien dong manh nhat (Std lon nhat): Q{quarterly['Std'].idxmax()}")
print(f"Quy co Range lon nhat: Q{quarterly['Range'].idxmax()}")

plt.figure(figsize=(9, 5))
plt.bar(quarterly.index, quarterly["Mean"], color="darkorange", edgecolor="black")
plt.xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
plt.xlabel("Quy")
plt.ylabel("Gia dong cua trung binh (VND)")
plt.title("Gia dong cua trung binh theo quy")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("cau2_gia_trung_binh_theo_quy.png", dpi=200)
plt.show()


# =========================
# 5. CAU 3 - NGAY GIAO DICH BAT THUONG
# =========================
abnormal_rows = []

for date, row in df.iterrows():
    # Ngay dau tien khong co gia tham chieu
    if pd.isna(row["Ref"]):
        continue

    ref = row["Ref"]
    floor_price = row["Floor"]
    ceiling_price = row["Ceiling"]

    for col in PRICE_COLUMNS:
        value = row[col]

        if np.isclose(value, floor_price, rtol=0, atol=1e-6):
            abnormal_type = "Gia san"
        elif np.isclose(value, ceiling_price, rtol=0, atol=1e-6):
            abnormal_type = "Gia tran"
        elif np.isclose(value, ref, rtol=0, atol=1e-6):
            abnormal_type = "Gia tham chieu"
        else:
            continue

        abnormal_rows.append({
            "Ngay": date.strftime("%Y-%m-%d"),
            "Moc_gia": col,
            "Loai_bat_thuong": abnormal_type,
            "Gia_tri": value,
            "Gia_tham_chieu": ref,
            "Gia_san": floor_price,
            "Gia_tran": ceiling_price
        })

abnormal_df = pd.DataFrame(abnormal_rows)

print("\n========== CAU 3: CAC MOC GIA BAT THUONG ==========")
if abnormal_df.empty:
    print("Khong tim thay moc gia bang gia san, gia tran hoac gia tham chieu.")
else:
    print(abnormal_df.to_string(index=False))
    abnormal_df.to_csv("cau3_ngay_bat_thuong.csv", index=False, encoding="utf-8-sig")


# =========================
# 6. CAU 4 - HISTOGRAM HAI NUA NAM
# =========================
first_half = df.loc[df["Quarter"].isin([1, 2]), "Close"].dropna()
second_half = df.loc[df["Quarter"].isin([3, 4]), "Close"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].hist(first_half, bins=25, color="royalblue", edgecolor="black", alpha=0.8)
axes[0].set_title("Phan bo gia Close - Q1 va Q2")
axes[0].set_xlabel("Gia dong cua (VND)")
axes[0].set_ylabel("Tan suat")
axes[0].grid(axis="y", alpha=0.2)

axes[1].hist(second_half, bins=25, color="darkorange", edgecolor="black", alpha=0.8)
axes[1].set_title("Phan bo gia Close - Q3 va Q4")
axes[1].set_xlabel("Gia dong cua (VND)")
axes[1].grid(axis="y", alpha=0.2)

fig.suptitle("So sanh phan bo gia trong hai nua nam")
plt.tight_layout()
plt.savefig("cau4_histogram_hai_nua_nam.png", dpi=200)
plt.show()


# =========================
# 7. CAU 5 - XU HUONG CA NAM VA BIEN DO LON NHAT
# =========================
df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

max_daily_range_date = df["Daily_Range"].idxmax()
max_daily_range = df.loc[max_daily_range_date, "Daily_Range"]
max_daily_high = df.loc[max_daily_range_date, PRICE_COLUMNS].max()
max_daily_low = df.loc[max_daily_range_date, PRICE_COLUMNS].min()

annual_high = df[PRICE_COLUMNS].max().max()
annual_low = df[PRICE_COLUMNS].min().min()
annual_range = annual_high - annual_low
annual_high_date = df[PRICE_COLUMNS].max(axis=1).idxmax()
annual_low_date = df[PRICE_COLUMNS].min(axis=1).idxmin()

print("\n========== CAU 5: TONG KET CA NAM ==========")
print(f"Gia cao nhat trong nam: {annual_high:.2f} VND, ngay {annual_high_date.date()}")
print(f"Gia thap nhat trong nam: {annual_low:.2f} VND, ngay {annual_low_date.date()}")
print(f"Bien do toan nam (cao nhat - thap nhat): {annual_range:.2f} VND")
print(f"Bien do trong ngay lon nhat: {max_daily_range:.2f} VND")
print(f"Ngay xay ra: {max_daily_range_date.date()}")
print(f"Gia cao/thap trong ngay do: {max_daily_high:.2f} / {max_daily_low:.2f} VND")

plt.figure(figsize=(13, 5))
plt.plot(df.index, df["Close"], label="Close", linewidth=1.2)
plt.plot(df.index, df["MA20"], label="MA20", linewidth=2)
plt.xlabel("Ngay")
plt.ylabel("Gia (VND)")
plt.title("Xu huong gia dong cua trong ca nam")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("cau5_xu_huong_ca_nam.png", dpi=200)
plt.show()


# =========================
# 8. ANIMATION - HAI BIEU DO NHU HINH MAU
# =========================
from matplotlib.collections import LineCollection

# === ĐƯA HÀM NÀY LÊN TRƯỚC VÒNG LẶP DÙNG NÓ ===
def get_price_color(value, ref, floor_price, ceiling_price):
    """Xac dinh mau gia theo gia tham chieu cua ngay."""
    if pd.isna(ref):
        return COLOR_REFERENCE
    if np.isclose(value, floor_price, rtol=0, atol=1e-6):
        return COLOR_FLOOR
    if np.isclose(value, ceiling_price, rtol=0, atol=1e-6):
        return COLOR_CEILING
    if np.isclose(value, ref, rtol=0, atol=1e-6):
        return COLOR_REFERENCE
    if value < ref:
        return COLOR_DOWN
    return COLOR_UP

n = len(df)
x = np.arange(n)
all_prices = df[PRICE_COLUMNS].to_numpy(dtype=float)

y_min = min(0, np.nanmin(all_prices) * 0.95)
y_max = np.nanmax(all_prices) * 1.08

fig, (ax_candle, ax_lines) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={"height_ratios": [1, 1]}
)
fig.patch.set_facecolor("black")

for ax in (ax_candle, ax_lines):
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")
    ax.grid(color="#555555", alpha=0.25)

ax_candle.set_ylabel("VND", color="white")
ax_candle.set_title("MO PHONG NEN GIA CHUNG KHOAN", color="white")
ax_lines.set_ylabel("VND", color="white")
ax_lines.set_xlabel("Ngay", color="white")
ax_lines.set_title("DIEN BIEN 7 MOC GIA TRONG NGAY", color="white")

ax_candle.set_ylim(y_min, y_max)
ax_lines.set_ylim(y_min, y_max)
ax_lines.set_xlim(-1, n)

# Tao san du lieu nen: moi ngay gom 1 bong nen va 1 than nen
wick_segments = []
body_segments = []
day_colors = []

for day in range(n):
    row = df.iloc[day]
    values = row[PRICE_COLUMNS].to_numpy(dtype=float)
    open_price, close_price = values[0], values[-1]
    high_price, low_price = np.max(values), np.min(values)

    wick_segments.append([(day, low_price), (day, high_price)])
    body_segments.append([
        (day, min(open_price, close_price)),
        (day, max(open_price, close_price))
    ])

    color = get_price_color(
        close_price, row["Ref"], row["Floor"], row["Ceiling"]
    )
    day_colors.append(color)

# Hai LineCollection: bong nen va than nen
wick_collection = LineCollection(
    [], colors=[], linewidths=1.0, zorder=2
)
body_collection = LineCollection(
    [], colors=[], linewidths=7.0, zorder=3
)
ax_candle.add_collection(wick_collection)
ax_candle.add_collection(body_collection)

# 7 duong gia
line_colors = ["#FFFFFF", "#A0A0A0", "#A0A0A0", "#A0A0A0",
               "#A0A0A0", "#A0A0A0", "#FFFFFF"]
line_artists = []
for i, col in enumerate(PRICE_COLUMNS):
    line, = ax_lines.plot(
        [], [], linewidth=1.0, alpha=0.85,
        color=line_colors[i], label=col
    )
    line_artists.append(line)

# Mot scatter duy nhat cho tat ca diem gia o bieu do duoi
point_scatter = ax_lines.scatter([], [], s=5, zorder=4)

ref_line, = ax_lines.plot([], [], color=COLOR_REFERENCE,
                          linewidth=0.8, linestyle="--", label="Tham chieu")
floor_line, = ax_lines.plot([], [], color=COLOR_FLOOR,
                            linewidth=0.8, linestyle=":", label="Gia san")
ceiling_line, = ax_lines.plot([], [], color=COLOR_CEILING,
                              linewidth=0.8, linestyle=":", label="Gia tran")

current_text = ax_candle.text(
    0.99, 0.93, "", transform=ax_candle.transAxes,
    color="white", ha="right", va="top", fontsize=10
)
price_text = ax_lines.text(
    0.99, 0.93, "", transform=ax_lines.transAxes,
    color="white", ha="right", va="top", fontsize=9
)

ax_lines.legend(
    loc="upper left", fontsize=7, facecolor="black",
    edgecolor="white", labelcolor="white"
)


def get_price_color(value, ref, floor_price, ceiling_price):
    """Xac dinh mau gia theo gia tham chieu cua ngay."""
    if pd.isna(ref):
        return COLOR_REFERENCE
    if np.isclose(value, floor_price, rtol=0, atol=1e-6):
        return COLOR_FLOOR
    if np.isclose(value, ceiling_price, rtol=0, atol=1e-6):
        return COLOR_CEILING
    if np.isclose(value, ref, rtol=0, atol=1e-6):
        return COLOR_REFERENCE
    if value < ref:
        return COLOR_DOWN
    return COLOR_UP


def init():
    wick_collection.set_segments([])
    body_collection.set_segments([])
    point_scatter.set_offsets(np.empty((0, 2)))

    for line in line_artists:
        line.set_data([], [])

    ref_line.set_data([], [])
    floor_line.set_data([], [])
    ceiling_line.set_data([], [])
    current_text.set_text("")
    price_text.set_text("")

    return ([
        wick_collection, body_collection, point_scatter,
        ref_line, floor_line, ceiling_line, current_text, price_text
    ] + line_artists)


def update(frame):
    # Hien thi nen tu ngay dau den ngay hien tai
    wick_collection.set_segments(wick_segments[:frame + 1])
    body_collection.set_segments(body_segments[:frame + 1])
    body_collection.set_color(day_colors[:frame + 1])
    wick_collection.set_color(day_colors[:frame + 1])

    # Cap nhat 7 duong gia
    for i, line in enumerate(line_artists):
        line.set_data(x[:frame + 1], all_prices[:frame + 1, i])

    # Cap nhat diem va mau cua 7 moc gia
    visible_points = all_prices[:frame + 1].reshape(-1)
    visible_x = np.repeat(x[:frame + 1], len(PRICE_COLUMNS))
    offsets = np.column_stack((visible_x, visible_points))

    visible_colors = []
    for day in range(frame + 1):
        row = df.iloc[day]
        for col in PRICE_COLUMNS:
            visible_colors.append(
                get_price_color(row[col], row["Ref"], row["Floor"], row["Ceiling"])
            )

    point_scatter.set_offsets(offsets)
    point_scatter.set_color(visible_colors)

    # Duong ngang tham chieu, san va tran cua ngay hien tai
    current_row = df.iloc[frame]
    if pd.notna(current_row["Ref"]):
        ref_line.set_data([0, n - 1], [current_row["Ref"], current_row["Ref"]])
        floor_line.set_data([0, n - 1], [current_row["Floor"], current_row["Floor"]])
        ceiling_line.set_data(
            [0, n - 1], [current_row["Ceiling"], current_row["Ceiling"]]
        )
    else:
        ref_line.set_data([], [])
        floor_line.set_data([], [])
        ceiling_line.set_data([], [])

    current_text.set_text(
        f"Ngay: {current_row.name.strftime('%d/%m/%Y')} | "
        f"Close: {current_row['Close']:,.0f} VND"
    )
    price_text.set_text(
        f"Open: {current_row['Open']:,.0f} | "
        f"Close: {current_row['Close']:,.0f} VND"
    )

    return ([
        wick_collection, body_collection, point_scatter,
        ref_line, floor_line, ceiling_line, current_text, price_text
    ] + line_artists)


ani = animation.FuncAnimation(
    fig, update, frames=n, init_func=init,
    interval=1000 / FPS, blit=False, repeat=False
)

plt.tight_layout()

try:
    writer = animation.FFMpegWriter(
        fps=FPS,
        metadata={"title": "Stock Price Animation"},
        bitrate=2200
    )
    ani.save(OUTPUT_VIDEO, writer=writer, dpi=DPI)
    print(f"\nDa xuat video thanh cong: {OUTPUT_VIDEO}")
except Exception as error:
    print("\nKhong the xuat MP4. Hay cai FFmpeg va them vao PATH.")
    print("Chi tiet loi:", error)

plt.show()
