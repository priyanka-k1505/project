import io
import os
import base64
from functools import wraps

import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image as PILImage
from torch_geometric.nn import GCNConv, global_mean_pool

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# ==================== DATABASE ====================
mydb = mysql.connector.connect(
    host="localhost", user="root", password="root", port="3306", database="kidney"
)
mycursor = mydb.cursor()


def _ensure_connection():
    mydb.ping(reconnect=True, attempts=3, delay=1)


def executionquery(query, values):
    _ensure_connection()
    mycursor.execute(query, values)
    mydb.commit()


def retrivequery1(query, values):
    _ensure_connection()
    mycursor.execute(query, values)
    return mycursor.fetchall()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_name"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ==================== AUTH / PAGES ====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]

        rows = retrivequery1("SELECT name, password FROM users WHERE LOWER(email) = LOWER(%s)", (email,))
        if not rows:
            return render_template("login.html", message="This email ID does not exist!")

        name, stored = rows[0]
        if not check_password_hash(stored, password):
            return render_template("login.html", message="Invalid password!")

        session["user_name"] = name
        session["user_email"] = email
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        c_password = request.form["c_password"]

        if password != c_password:
            return render_template("register.html", message="Confirm password does not match!")

        if retrivequery1("SELECT id FROM users WHERE LOWER(email) = LOWER(%s)", (email,)):
            return render_template("register.html", message="This email ID already exists!")

        executionquery(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, generate_password_hash(password)),
        )
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/performance")
@login_required
def performance():
    return render_template("performance.html")


# ==================== BEST MODEL: ASGCN-EffNetV2 ====================
# Architecture copied exactly from the notebook (99.15% test accuracy).
class ResidualGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.residual(x)
        x = self.gcn(x, edge_index)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_feat, graph_feat):
        q = local_feat.unsqueeze(1)
        kv = graph_feat.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(out.squeeze(1) + local_feat)


class ASGCN_EffNetV2(nn.Module):
    def __init__(self, num_classes=2, k_neighbors=8, hidden_dim=224, num_gcn_layers=2, dropout=0.30):
        super().__init__()
        # pretrained=False: real weights come from our checkpoint
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0, global_pool="")
        self.patch_dim = 1280
        self.spatial_size = 7
        self.k = k_neighbors
        self.proj = nn.Linear(self.patch_dim, hidden_dim)
        self.gcn_blocks = nn.ModuleList(
            [ResidualGCNBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_gcn_layers)]
        )
        self.local_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = CrossAttentionFusion(hidden_dim, num_heads=4)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def build_adaptive_graph(self, nodes, batch_size, num_patches):
        edge_index_list = []
        for b in range(batch_size):
            start = b * num_patches
            feats = F.normalize(nodes[start:start + num_patches], p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            _, topk_idx = torch.topk(sim, k=min(self.k, num_patches), dim=1)
            src = torch.arange(num_patches, device=nodes.device).unsqueeze(1).expand(-1, topk_idx.size(1)).reshape(-1)
            dst = topk_idx.reshape(-1)
            edge_index_list.append(torch.stack([src + start, dst + start], dim=0))
        return torch.cat(edge_index_list, dim=1)

    def forward(self, x):
        features = self.backbone(x)
        batch_size = features.shape[0]
        num_patches = self.spatial_size * self.spatial_size

        nodes = features.permute(0, 2, 3, 1).reshape(batch_size * num_patches, -1)
        nodes = self.proj(nodes)
        edge_index = self.build_adaptive_graph(nodes, batch_size, num_patches)

        h = nodes
        for block in self.gcn_blocks:
            h = block(h, edge_index)

        batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(num_patches)
        graph_feat = global_mean_pool(h, batch_vec)

        local_feat = self.local_pool(features).flatten(1)
        local_feat = self.proj(local_feat)

        fused = self.fusion(local_feat, graph_feat)
        return self.classifier(fused)


# ==================== LOAD MODEL ====================
MODEL_PATH = os.path.join("models", "asgcn-effnetv2_(novelty)_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# Read the architecture from the checkpoint itself, so it always matches
HIDDEN_DIM = state_dict["proj.weight"].shape[0]
NUM_GCN_LAYERS = len({k.split(".")[1] for k in state_dict if k.startswith("gcn_blocks.")})
K_NEIGHBORS = 6  # not stored in the weights (notebook text says 6, later code says 8)

model = ASGCN_EffNetV2(num_classes=2, k_neighbors=K_NEIGHBORS,
                       hidden_dim=HIDDEN_DIM, num_gcn_layers=NUM_GCN_LAYERS)
model.load_state_dict(state_dict)
model.to(device).eval()
print(f"Loaded ASGCN-EffNetV2 (hidden={HIDDEN_DIM}, gcn_layers={NUM_GCN_LAYERS}, k={K_NEIGHBORS})")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ImageFolder sorts classes alphabetically: ['Normal', 'stone']
CLASS_NAMES = ["Normal", "Stone"]


@app.route("/classification", methods=["GET", "POST"])
@login_required
def classification():
    ctx = dict(prediction=None, probs=None, image_display=None, error=None)

    if request.method == "POST":
        file = request.files.get("file")
        if file is None:
            ctx["error"] = "No file part in the request."
        elif file.filename == "":
            ctx["error"] = "No file selected."
        else:
            try:
                image = PILImage.open(file.stream).convert("RGB")

                thumb = image.copy()
                thumb.thumbnail((700, 700))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=90)
                ctx["image_display"] = base64.b64encode(buf.getvalue()).decode("utf-8")

                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

                ctx["probs"] = {name: round(p * 100, 2) for name, p in zip(CLASS_NAMES, probs)}
                ctx["prediction"] = CLASS_NAMES[int(max(range(len(probs)), key=probs.__getitem__))]
            except Exception as e:
                ctx["error"] = f"Could not process that image: {e}"

    return render_template("classification.html", **ctx)


@app.errorhandler(413)
def too_large(_):
    return render_template("classification.html", prediction=None, probs=None, image_display=None,
                           error="File is too large. Maximum size is 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)