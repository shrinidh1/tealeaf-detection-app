"""Disease content shown on the Disease Guide page and under the detection results.

Every entry has:
  title      heading of the card
  image      name of the picture in the assets/ folder (any of .jpg .jpeg .png .webp; see find_image() in app.py)
  cause      one italic line under the title
  summary    short intro paragraph
  details    optional extra notes (one item = a paragraph, several items = bullets)
  symptoms / spread / management    lists of short bullet points

get_info(name) accepts the class name from the model ("Brown Blight", "brown_blight", ...) in any
capitalisation and returns a copy of the matching entry, or None if there is none.
"""
import copy
import re

_MANAGEMENT_NOTE = "use only products and doses recommended by your local tea research institute."

DISEASES = {
    "Anthracnose": {
        "title": "Anthracnose",
        "image": "anthracnose",
        "cause": "Fungi of the genus Colletotrichum",
        "summary": (
            "A fungal leaf disease. In tea, the same fungal genus also causes brown blight, and in some "
            "regions the two names are used for similar-looking symptoms."
        ),
        "details": [
            "Primarily caused by fungal species in the genus Colletotrichum (such as Colletotrichum camelliae, "
            "Colletotrichum gloeosporioides, and Colletotrichum theae-sinensis) as well as Sinodiscula "
            "theae-sinensis (syn. Discula theae-sinensis)."
        ],
        "symptoms": [
            "Brown to dark-brown lesions, often starting at the leaf tip or margin, that can enlarge and merge.",
            "Older lesions may become grey or tan with a darker border.",
            "Tiny black fruiting bodies may appear in older lesions.",
        ],
        "spread": [
            "Spores spread by rain splash; warm, humid, wet weather, wounds and plant stress favour infection.",
        ],
        "management": [
            "Remove and destroy infected leaves and shoots.",
            "Avoid wounding the plant, and improve drainage and air circulation.",
            "Keep bushes well nourished.",
            "For chemical control, " + _MANAGEMENT_NOTE,
        ],
    },
    "Brown Blight": {
        "title": "Brown Blight",
        "image": "brown_blight",
        "cause": "Fungus: Colletotrichum camelliae (also given as Glomerella cingulata)",
        "summary": "A common foliar fungal disease in India, China, Sri Lanka, Africa and other tea-growing regions.",
        "details": [
            "Fungi: Primarily caused by species such as Colletotrichum (e.g., Colletotrichum acutatum, "
            "Colletotrichum camelliae) and Pestalotiopsis.",
            "These fungi are generally considered weak or opportunistic pathogens that attack tea bushes already "
            "weakened by poor nutrition, physical damage, pruning stress, or adverse weather.",
        ],
        "symptoms": [
            "Infection starts at the leaf margin and spreads inwards in patches.",
            "Patch edges are sharply defined, often with a delicate concentric zone.",
            "The upper surface is yellowish to chocolate brown at first, then turns grey from the centre outwards.",
            "Minute, black, scattered dot-like fruiting bodies of the fungus appear on both leaf surfaces.",
        ],
        "spread": [
            "Fungal spores spread by rain splash and wind; warm, humid, wet conditions favour infection.",
        ],
        "management": [
            "Remove and destroy badly infected leaves and shoots.",
            "Avoid wounding leaves during plucking and handling.",
            "Keep bushes well nourished and reduce long periods of leaf wetness (airflow, drainage).",
            "If fungicide is needed, " + _MANAGEMENT_NOTE,
        ],
    },
    "White Spot": {
        "title": "White Spot (White Scab)",
        "image": "white_spot",
        "cause": "Fungus: Elsinoe leucospila; Phyllosticta theifolia has also been reported",
        "summary": (
            "A fungal leaf disease of tea that mainly occurs in cool, humid, high-altitude tea areas. Recent "
            "research indicates that Elsinoe leucospila is the main cause, while Phyllosticta may infect tissue "
            "that is already diseased."
        ),
        "details": [],
        "symptoms": [
            "Small whitish to pale-grey spots, often many scattered over the leaf.",
            "Severe infection can damage leaves and lower tea quality (reduced caffeine and tea polyphenols "
            "have been reported).",
        ],
        "spread": [
            "Favoured by cool, humid, wet conditions, typical of high-altitude gardens.",
        ],
        "management": [
            "Remove and destroy infected leaves and shoots.",
            "Improve air circulation and avoid excess shade and long periods of leaf wetness.",
            "Maintain balanced nutrition to keep bushes vigorous.",
            "For chemical control, " + _MANAGEMENT_NOTE,
        ],
    },
    "Algal Leaf Spot": {
        "title": "Algal Leaf Spot",
        "image": "algal_leaf_spot",
        "cause": "Alga: Cephaleuros virescens",
        "summary": (
            "A leaf disease caused by an alga rather than a fungus. It is generally considered a minor disease, "
            "and usually does not seriously harm the plant."
        ),
        "details": [
            "Algal leaf spot in tea, commonly known as red rust, is a plant disease caused by the parasitic green "
            "alga Cephaleuros (primarily species like Cephaleuros parasiticus or Cephaleuros virescens). Unlike "
            "most common tea diseases caused by fungi or bacteria, this is one of the few parasitic algal "
            "afflictions affecting crops."
        ],
        "symptoms": [
            "Roughly circular, raised lesions that are purple to reddish-brown.",
            "Rust-coloured, spore-like bodies on the leaf surface give the spots a reddish tinge.",
            "Most spots develop on the upper leaf surface.",
            "Older infections become greenish-grey and look like lichen.",
        ],
        "spread": [
            "Spores are spread by wind and rain, and the alga can spread from leaves to branches and fruit.",
            "Favoured by poor soil drainage, imbalanced nutrition, and relatively high temperature and humidity.",
        ],
        "management": [
            "Avoid plant stress and poorly drained sites.",
            "Improve air circulation in the canopy to reduce humidity and how long leaves stay wet.",
            "Improve soil nutrition with balanced nitrogen, phosphorus and potassium.",
            "Remove and destroy affected plant parts.",
            "Copper-based sprays such as Bordeaux mixture are used in some guidelines. Follow the dose and timing "
            "recommended by your local tea research institute.",
        ],
    },
}

# other names a class might be called by
_ALIASES = {
    "whitescab": "White Spot",
    "algalspot": "Algal Leaf Spot",
    "algalleafspot": "Algal Leaf Spot",
    "redrust": "Algal Leaf Spot",
}


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_LOOKUP = {_norm(k): k for k in DISEASES}
_LOOKUP.update({_norm(v["title"]): k for k, v in DISEASES.items()})
_LOOKUP.update(_ALIASES)


def get_info(name):
    """Return a copy of the entry for a disease name, or None if unknown."""
    key = _LOOKUP.get(_norm(name))
    return copy.deepcopy(DISEASES[key]) if key else None