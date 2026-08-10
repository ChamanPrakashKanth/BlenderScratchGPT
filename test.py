import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class MassSpringDamper(nn.Module):

    def __init__(self, m, c, k):
        super().__init__()

        # Physical parameters
        self.m = m
        self.c = c
        self.k = k

        # Neural network: t -> x(t)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

    def equation(self, t):

        # x(t)
        x = self.forward(t)

        # dx/dt
        x_dot = torch.autograd.grad(
            x,
            t,
            grad_outputs=torch.ones_like(x),
            create_graph=True
        )[0]

        # d²x/dt²
        x_ddot = torch.autograd.grad(
            x_dot,
            t,
            grad_outputs=torch.ones_like(x_dot),
            create_graph=True
        )[0]

        # m*x'' + c*x' + k*x = 0
        residual = (
            self.m * x_ddot
            + self.c * x_dot
            + self.k * x
        )

        return residual


# --------------------------------------------------
# Physical system
# --------------------------------------------------

model = MassSpringDamper(
    m=1.0,
    c=0.5,
    k=4.0
)


# --------------------------------------------------
# Time points
# --------------------------------------------------

t = torch.linspace(0, 10, 500).reshape(-1, 1)
t.requires_grad_(True)


# --------------------------------------------------
# Initial condition
# x(0) = 1
# x'(0) = 0
# --------------------------------------------------

t0 = torch.zeros(1, 1, requires_grad=True)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(5000):

    optimizer.zero_grad()

    # Physics residual
    residual = model.equation(t)

    # Physics loss
    physics_loss = torch.mean(residual ** 2)

    # Initial displacement
    x0 = model(t0)

    # Initial velocity
    x0_dot = torch.autograd.grad(
        x0,
        t0,
        grad_outputs=torch.ones_like(x0),
        create_graph=True
    )[0]

    # Initial-condition loss
    initial_loss = (
        (x0 - 1.0) ** 2
        + (x0_dot - 0.0) ** 2
    ).mean()

    # Total loss
    loss = physics_loss + initial_loss

    # Backpropagation
    loss.backward()

    # Update neural-network weights
    optimizer.step()

    if epoch % 500 == 0:

        print(
            f"Epoch: {epoch:4d} | "
            f"Total Loss: {loss.item():.6e} | "
            f"Physics Loss: {physics_loss.item():.6e} | "
            f"Initial Loss: {initial_loss.item():.6e}"
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    x_pred = model(t)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.plot(t.detach().numpy(), x_pred.numpy())

plt.xlabel("Time")
plt.ylabel("Displacement x(t)")
plt.title("PINN: Mass-Spring-Damper")

plt.grid()
plt.show()