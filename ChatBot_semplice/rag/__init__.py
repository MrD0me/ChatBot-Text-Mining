"""Chatbot RAG su PDF: estrazione, indice ibrido, generazione via API."""

try:
    # Su alcuni PC Windows (antivirus che ispeziona HTTPS, proxy) Python non
    # riconosce i certificati che Windows invece accetta. truststore fa usare
    # a Python l'archivio certificati del sistema operativo, come fanno pip e curl.
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - truststore è opzionale
    pass
