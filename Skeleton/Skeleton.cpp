//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Par�j Bal�zs
// Neptun : E1512Q
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
struct Complex {
	//--------------------------
	float x, y;

	Complex(float x0 = 0, float y0 = 0) { x = x0, y = y0; }
	Complex operator+(Complex r) { return Complex(x + r.x, y + r.y); }
	Complex operator-(Complex r) { return Complex(x - r.x, y - r.y); }
	Complex operator*(Complex r) { return Complex(x * r.x - y * r.y, x * r.y + y * r.x); }
	Complex operator/(Complex r) {
		float l = r.x * r.x + r.y * r.y;
		return (*this) * Complex(r.x / l, -r.y / l);
	}
};

Complex Polar(float r, float phi) {
	return Complex(r * cosf(phi), r * sinf(phi));
}
mat4 TransposeMatrix(const mat4 A) {
	mat4 B;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			B.m[i][j] = A.m[j][i];
	return B;
}
mat4 InvertMatrix(const mat4 m) //source: mesa3d.org, frameworkhoz alakitva
{
	mat4 invOut;
	float inv[16], det;
	int i;

	inv[0] = m.m[1][1] * m.m[2][2] * m.m[3][3] -
		m.m[1][1] * m.m[2][3] * m.m[3][2] -
		m.m[2][1] * m.m[1][2] * m.m[3][3] +
		m.m[2][1] * m.m[1][3] * m.m[3][2] +
		m.m[3][1] * m.m[1][2] * m.m[2][3] -
		m.m[3][1] * m.m[1][3] * m.m[2][2];

	inv[4] = -m.m[1][0] * m.m[2][2] * m.m[3][3] +
		m.m[1][0] * m.m[2][3] * m.m[3][2] +
		m.m[2][0] * m.m[1][2] * m.m[3][3] -
		m.m[2][0] * m.m[1][3] * m.m[3][2] -
		m.m[3][0] * m.m[1][2] * m.m[2][3] +
		m.m[3][0] * m.m[1][3] * m.m[2][2];

	inv[8] = m.m[1][0] * m.m[2][1] * m.m[3][3] -
		m.m[1][0] * m.m[2][3] * m.m[3][1] -
		m.m[2][0] * m.m[1][1] * m.m[3][3] +
		m.m[2][0] * m.m[1][3] * m.m[3][1] +
		m.m[3][0] * m.m[1][1] * m.m[2][3] -
		m.m[3][0] * m.m[1][3] * m.m[2][1];

	inv[12] = -m.m[1][0] * m.m[2][1] * m.m[3][2] +
		m.m[1][0] * m.m[2][2] * m.m[3][1] +
		m.m[2][0] * m.m[1][1] * m.m[3][2] -
		m.m[2][0] * m.m[1][2] * m.m[3][1] -
		m.m[3][0] * m.m[1][1] * m.m[2][2] +
		m.m[3][0] * m.m[1][2] * m.m[2][1];

	inv[1] = -m.m[0][1] * m.m[2][2] * m.m[3][3] +
		m.m[0][1] * m.m[2][3] * m.m[3][2] +
		m.m[2][1] * m.m[0][2] * m.m[3][3] -
		m.m[2][1] * m.m[0][3] * m.m[3][2] -
		m.m[3][1] * m.m[0][2] * m.m[2][3] +
		m.m[3][1] * m.m[0][3] * m.m[2][2];

	inv[5] = m.m[0][0] * m.m[2][2] * m.m[3][3] -
		m.m[0][0] * m.m[2][3] * m.m[3][2] -
		m.m[2][0] * m.m[0][2] * m.m[3][3] +
		m.m[2][0] * m.m[0][3] * m.m[3][2] +
		m.m[3][0] * m.m[0][2] * m.m[2][3] -
		m.m[3][0] * m.m[0][3] * m.m[2][2];

	inv[9] = -m.m[0][0] * m.m[2][1] * m.m[3][3] +
		m.m[0][0] * m.m[2][3] * m.m[3][1] +
		m.m[2][0] * m.m[0][1] * m.m[3][3] -
		m.m[2][0] * m.m[0][3] * m.m[3][1] -
		m.m[3][0] * m.m[0][1] * m.m[2][3] +
		m.m[3][0] * m.m[0][3] * m.m[2][1];

	inv[13] = m.m[0][0] * m.m[2][1] * m.m[3][2] -
		m.m[0][0] * m.m[2][2] * m.m[3][1] -
		m.m[2][0] * m.m[0][1] * m.m[3][2] +
		m.m[2][0] * m.m[0][2] * m.m[3][1] +
		m.m[3][0] * m.m[0][1] * m.m[2][2] -
		m.m[3][0] * m.m[0][2] * m.m[2][1];

	inv[2] = m.m[0][1] * m.m[1][2] * m.m[3][3] -
		m.m[0][1] * m.m[1][3] * m.m[3][2] -
		m.m[1][1] * m.m[0][2] * m.m[3][3] +
		m.m[1][1] * m.m[0][3] * m.m[3][2] +
		m.m[3][1] * m.m[0][2] * m.m[1][3] -
		m.m[3][1] * m.m[0][3] * m.m[1][2];

	inv[6] = -m.m[0][0] * m.m[1][2] * m.m[3][3] +
		m.m[0][0] * m.m[1][3] * m.m[3][2] +
		m.m[1][0] * m.m[0][2] * m.m[3][3] -
		m.m[1][0] * m.m[0][3] * m.m[3][2] -
		m.m[3][0] * m.m[0][2] * m.m[1][3] +
		m.m[3][0] * m.m[0][3] * m.m[1][2];

	inv[10] = m.m[0][0] * m.m[1][1] * m.m[3][3] -
		m.m[0][0] * m.m[1][3] * m.m[3][1] -
		m.m[1][0] * m.m[0][1] * m.m[3][3] +
		m.m[1][0] * m.m[0][3] * m.m[3][1] +
		m.m[3][0] * m.m[0][1] * m.m[1][3] -
		m.m[3][0] * m.m[0][3] * m.m[1][1];

	inv[14] = -m.m[0][0] * m.m[1][1] * m.m[3][2] +
		m.m[0][0] * m.m[1][2] * m.m[3][1] +
		m.m[1][0] * m.m[0][1] * m.m[3][2] -
		m.m[1][0] * m.m[0][2] * m.m[3][1] -
		m.m[3][0] * m.m[0][1] * m.m[1][2] +
		m.m[3][0] * m.m[0][2] * m.m[1][1];

	inv[3] = -m.m[0][1] * m.m[1][2] * m.m[2][3] +
		m.m[0][1] * m.m[1][3] * m.m[2][2] +
		m.m[1][1] * m.m[0][2] * m.m[2][3] -
		m.m[1][1] * m.m[0][3] * m.m[2][2] -
		m.m[2][1] * m.m[0][2] * m.m[1][3] +
		m.m[2][1] * m.m[0][3] * m.m[1][2];

	inv[7] = m.m[0][0] * m.m[1][2] * m.m[2][3] -
		m.m[0][0] * m.m[1][3] * m.m[2][2] -
		m.m[1][0] * m.m[0][2] * m.m[2][3] +
		m.m[1][0] * m.m[0][3] * m.m[2][2] +
		m.m[2][0] * m.m[0][2] * m.m[1][3] -
		m.m[2][0] * m.m[0][3] * m.m[1][2];

	inv[11] = -m.m[0][0] * m.m[1][1] * m.m[2][3] +
		m.m[0][0] * m.m[1][3] * m.m[2][1] +
		m.m[1][0] * m.m[0][1] * m.m[2][3] -
		m.m[1][0] * m.m[0][3] * m.m[2][1] -
		m.m[2][0] * m.m[0][1] * m.m[1][3] +
		m.m[2][0] * m.m[0][3] * m.m[1][1];

	inv[15] = m.m[0][0] * m.m[1][1] * m.m[2][2] -
		m.m[0][0] * m.m[1][2] * m.m[2][1] -
		m.m[1][0] * m.m[0][1] * m.m[2][2] +
		m.m[1][0] * m.m[0][2] * m.m[2][1] +
		m.m[2][0] * m.m[0][1] * m.m[1][2] -
		m.m[2][0] * m.m[0][2] * m.m[1][1];

	det = m.m[0][0] * inv[0] + m.m[0][1] * inv[4] + m.m[0][2] * inv[8] + m.m[0][3] * inv[12];
	if (det == 0)
		return m;

	det = 1.0 / det;

	invOut = { inv[0] * det,inv[1] * det,inv[2] * det, inv[3] * det,
				inv[4] * det, inv[5] * det, inv[6] * det, inv[7] * det,
				inv[8] * det, inv[9] * det, inv[10] * det, inv[11] * det,
				inv[12] * det, inv[13] * det, inv[14] * det, inv[15] * det };
	return invOut;
}
// Bezier using Bernstein polynomials
class BezierCurve {
	std::vector<vec4> wCtrlPoints;

	float B(int i, float t) {
		int n = wCtrlPoints.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}
public:
	void AddControlPoint(vec4 cPoint) {
		vec4 wVertex = vec4(cPoint.x, cPoint.y, cPoint.z, 1);
		wCtrlPoints.push_back(wVertex);
	}
	vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 1);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++) wPoint += wCtrlPoints[n] * B(n, t);
		return wPoint;
	}
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }
	std::vector<vec4> TesselatedPoints() {
		std::vector<vec4> vertexData;
		if (wCtrlPoints.size() >= 2) {	// draw curve

			for (int i = 0; i < 100; i++) {	// Tessellate
				float tNormalized = (float)i / (100 - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				vertexData.push_back(wVertex);
			}
		}
		return vertexData;
	}
};

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

mat4 transformMatrix(mat4 t, mat4 r, mat4 s) {
	mat4 result;
	result = s * r*t;
	return result;
}
inline float dotv4(const vec4& v1, const vec4& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w); }

struct Hyperboloid : public Intersectable {

	Hyperboloid(Material* _material) {
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		mat4 trans = transformMatrix(TranslateMatrix(vec3(0, 0, 0)), RotationMatrix(M_PI_4, vec3(1, 0, 0)), ScaleMatrix(vec3(1, 1, 1)));
		mat4 transinv = InvertMatrix(trans);
		mat4 transinvt = TransposeMatrix(transinv);
		mat4 cyl = { 1,0,0,0,
					   0,1,0,0,
					   0,0,-1,0,
					   0,0,0,-1 };

		mat4 Q = transinv * cyl*transinvt;
		vec4 dir(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 start(ray.start.x, ray.start.y, ray.start.z, 1);
		float a = dotv4(dir*Q, dir);
		float b = dotv4(start*Q, dir) + dotv4(dir*Q, start);
		float c = dotv4(start*Q, start) - 1;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) {
			return hit;
		}
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) {
			return hit;
		}
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		float dx = 2 * hit.position.x * Q.m[0][0] + hit.position.y * Q.m[1][0] + hit.position.z * Q.m[2][0] + Q.m[3][0] +
			hit.position.y * Q.m[0][1] + hit.position.z * Q.m[0][2] + Q.m[0][3];
		float dy = 2 * hit.position.y * Q.m[1][1] + hit.position.x * Q.m[0][1] + hit.position.z * Q.m[2][1] + Q.m[3][1] +
			hit.position.x * Q.m[1][0] + hit.position.z * Q.m[1][2] + Q.m[1][3];
		float dz = 2 * hit.position.z * Q.m[2][2] + hit.position.x * Q.m[0][2] + hit.position.y * Q.m[1][2] + Q.m[3][2] +
			hit.position.x * Q.m[2][0] + hit.position.y * Q.m[2][1] + Q.m[2][3];
		hit.normal = normalize(vec3(dx, dy, dz));
		hit.material = material;
		return hit;
	}
};
struct Cylinder : public Intersectable {

	Cylinder(Material* _material) {
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		mat4 trans = transformMatrix(TranslateMatrix(vec3(25, 15, 0)), RotationMatrix(M_PI_4, vec3(1, 0, 0)), ScaleMatrix(vec3(1, 1, 1)));
		mat4 transinv = InvertMatrix(trans);
		mat4 transinvt = TransposeMatrix(transinv);
		mat4 cyl = { 1,0,0,0,
					   0,1,0,0,
					   0,0,0,0,
					   0,0,0,-1 };

		mat4 Q = transinv * cyl*transinvt;
		vec4 dir(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 start(ray.start.x, ray.start.y, ray.start.z, 1);
		float a = dotv4(dir*Q, dir);
		float b = dotv4(start*Q, dir) + dotv4(dir*Q, start);
		float c = dotv4(start*Q, start);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) {
			return hit;
		}
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) {
			return hit;
		}
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		float dx = 2 * hit.position.x * Q.m[0][0] + hit.position.y * Q.m[1][0] + hit.position.z * Q.m[2][0] + Q.m[3][0] +
			hit.position.y * Q.m[0][1] + hit.position.z * Q.m[0][2] + Q.m[0][3];
		float dy = 2 * hit.position.y * Q.m[1][1] + hit.position.x * Q.m[0][1] + hit.position.z * Q.m[2][1] + Q.m[3][1] +
			hit.position.x * Q.m[1][0] + hit.position.z * Q.m[1][2] + Q.m[1][3];
		float dz = 2 * hit.position.z * Q.m[2][2] + hit.position.x * Q.m[0][2] + hit.position.y * Q.m[1][2] + Q.m[3][2] +
			hit.position.x * Q.m[2][0] + hit.position.y * Q.m[2][1] + Q.m[2][3];
		hit.normal = normalize(vec3(dx, dy, dz));
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};
const float epsilon = 0.1f;
struct Light {
	vec3 direction; //ez kb source lenne, csak nem akartam atirni mindenhol
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = (_direction);
		Le = _Le;

	}
};

struct CoherentLights {
	std::vector<vec4> points;
	CoherentLights(std::vector<vec4> lights) {
		points = lights;
	}
	std::vector<vec4> getPoints() {
		return points;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }



class Scene {
public:
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	CoherentLights* cl;
	Camera camera;
	vec3 La;

	void build() {
		vec3 eye = vec3(30, 0, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		BezierCurve bc;
		bc.AddControlPoint(vec4(1.8, -4, 0, 1));
		bc.AddControlPoint(vec4(0.6, 0, 0, 1));
		bc.AddControlPoint(vec4(-0.6, 0, 0, 1));
		bc.AddControlPoint(vec4(-1.8, -4, 0, 1));
		std::vector<vec4> curvepoints = bc.TesselatedPoints();
		mat4 trans = transformMatrix(TranslateMatrix(vec3(0, -2, 0)), RotationMatrix(-M_PI_2, vec3(1, 0, 0)), ScaleMatrix(vec3(1, 1, 1)));
		for (int i = 0; i < curvepoints.size(); i++) { //raillesztes a hengerre
			curvepoints[i] = curvepoints[i] * trans;
			vec2 point = normalize(vec2(curvepoints[i].x, curvepoints[i].y));
			curvepoints[i].x = point.x;
			curvepoints[i].y = point.y;
		}
		La = vec3(0.4f, 0.4f, 0.6f);
		vec3 lightDirection(10, -5, -3), Le(10, 10, 10);
		lights.push_back(new Light(lightDirection, Le));
		mat4 trans2 = transformMatrix(TranslateMatrix(vec3(0, 0, 0)), RotationMatrix((float)-10 * M_PI / (float)180, vec3(0, 0, 1)), ScaleMatrix(vec3(1, 1, 1)));
		mat4 trans3 = transformMatrix(TranslateMatrix(vec3(25, 15, 0)), RotationMatrix(M_PI_4, vec3(1, 0, 0)), ScaleMatrix(vec3(1, 1, 1)));
		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material * material = new Material(kd, ks, 50);
		Material * mat2 = new Material(vec3(1, 0, 0), vec3(2, 2, 2), 50);
		for (int i = 0; i < curvepoints.size(); i++) {
			curvepoints[i] = curvepoints[i] * trans2;
			curvepoints[i] = curvepoints[i] * trans3;
		}

		objects.push_back(new Hyperboloid(material));
		objects.push_back(new Cylinder(material));
		cl = new CoherentLights(curvepoints);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) { //valamiert nem az igazi, ha mindket fele fenyforras benne van
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			vec3 outDir = normalize(light->direction - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, outDir);

			float d = length(hit.position - light->direction);
			float cosTheta = dot(hit.normal, outDir);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le / (d*d) * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + outDir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le / (d*d)* hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		Complex sp;
		for (int i = 0; i < cl->points.size(); i++) {
			vec3 convert(cl->points[i].x, cl->points[i].y, cl->points[i].z);
			float d = length(hit.position - convert);
			float intens;
			if (i == cl->points.size() - 1) intens = length(convert - vec3(cl->points[i - 1].x, cl->points[i - 1].y, cl->points[i - 1].z));
			else intens = length(-convert + vec3(cl->points[i + 1].x, cl->points[i + 1].y, cl->points[i + 1].z));
			float amplitude = 100;
			float angle = 2 * M_PI / 0.526*d;
			sp = sp + Polar(amplitude*intens / d, angle);
			vec3 outDir = normalize(convert - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, outDir);

			float cosTheta = dot(hit.normal, outDir);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + (sp.x*sp.x)*vec3(0, 1, 0) / (d*d) * hit.material->kd * cosTheta; //szerintem sp.x^2+sp.y^2 kellene de azzal csak egy zold paca jelenik meg
				vec3 halfway = normalize(-ray.dir + outDir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + (sp.x*sp.x + sp.y*sp.y)*vec3(0, 1, 0) / (d*d)* hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}

		return outRadiance;
	}

};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}