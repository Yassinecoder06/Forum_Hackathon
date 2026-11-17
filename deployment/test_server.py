import base64

from fastapi.testclient import TestClient

from deployment.server import create_app


def test_preview_protects_api_and_frontend_and_preserves_routes(tmp_path):
    (tmp_path/'index.html').write_text('<html>SignaCulture</html>')
    app = create_app(tmp_path, 'test-user', 'test-password')
    client = TestClient(app)
    assert client.get('/healthz').status_code == 200
    assert client.get('/').status_code == 401
    assert client.get('/api/library').status_code == 401
    assert client.delete('/api/library/example').status_code == 401
    assert client.get('/api/health', headers={'Authorization': 'Basic invalid'}).status_code == 401
    assert client.get('/', auth=('test-user', 'wrong')).status_code == 401
    assert client.get('/api/health', auth=('test-user', 'test-password')).json()['status'] == 'ok'
    assert 'SignaCulture' in client.get('/studio', auth=('test-user', 'test-password')).text
    assert client.get('/missing.js', auth=('test-user', 'test-password')).status_code == 404
    assert client.get('/api/does-not-exist', auth=('test-user', 'test-password')).status_code == 404
