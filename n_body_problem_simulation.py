import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import copy
import time


class Body:
    """
    Represents a stellar body with mass, position, and velocity in 3D space.
    """

    def __init__(
        self, mass: float, position: np.ndarray, velocity: np.ndarray, name: str = ""
    ):
        """
        Initialize a stellar body.

        Args:
            mass: Mass of the body (kg)
            position: 3D position vector (m)
            velocity: 3D velocity vector (m/s)
            name: Optional name for the body
        """
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.name = name

        # Store trajectory for visualization
        self.trajectory = [self.position.copy()]

    def gravitational_force(self, other: "Body", G: float = 6.67430e-11) -> np.ndarray:
        """
        Calculate gravitational force exerted by another body on this body.

        Args:
            other: The other body
            G: Gravitational constant (m³/kg/s²)

        Returns:
            Force vector (N)
        """
        # Vector from this body to the other body
        r_vec = other.position - self.position
        r_magnitude = np.linalg.norm(r_vec)

        # Avoid division by zero and numerical instability
        if r_magnitude < 1e-10:
            return np.zeros(3)

        # Newton's law of universal gravitation: F = G * m1 * m2 / r²
        force_magnitude = G * self.mass * other.mass / (r_magnitude**2)

        # Unit vector in direction of force
        r_unit = r_vec / r_magnitude

        return force_magnitude * r_unit

    def update_acceleration(self, bodies: List["Body"], G: float = 6.67430e-11):
        """
        Update acceleration based on gravitational forces from all other bodies.

        Args:
            bodies: List of all bodies in the system
            G: Gravitational constant
        """
        self.acceleration = np.zeros(3)

        for other in bodies:
            if other is not self:
                force = self.gravitational_force(other, G)
                # F = ma, so a = F/m
                self.acceleration += force / self.mass

    def update_position_velocity(self, dt: float):
        """
        Update position and velocity using Verlet integration.

        Args:
            dt: Time step (s)
        """
        # Verlet integration for better stability
        # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        # v(t+dt) = v(t) + a(t)*dt

        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt

        # Store trajectory point
        self.trajectory.append(self.position.copy())


class NBodySimulation:
    """
    Manages the N-body gravitational simulation.
    """

    def __init__(self, bodies: List[Body], G: float = 6.67430e-11):
        """
        Initialize the simulation.

        Args:
            bodies: List of Body objects
            G: Gravitational constant
        """
        self.bodies = bodies
        self.G = G
        self.time = 0.0
        self.dt = 0.0

    def step(self, dt: float):
        """
        Perform one simulation step using Leapfrog integration.

        Args:
            dt: Time step (s)
        """
        self.dt = dt

        # Calculate accelerations for all bodies
        for body in self.bodies:
            body.update_acceleration(self.bodies, self.G)

        # Update positions and velocities
        for body in self.bodies:
            body.update_position_velocity(dt)

        self.time += dt

    def simulate(self, duration: float, dt: float):
        """
        Run the simulation for a specified duration.

        Args:
            duration: Total simulation time (s)
            dt: Time step (s)
        """
        steps = int(duration / dt)

        for i in range(steps):
            self.step(dt)

            if i % 1000 == 0:
                print(f"Step {i}/{steps}, Time: {self.time:.2e} s")

    def total_energy(self) -> float:
        """
        Calculate total energy of the system (kinetic + potential).

        Returns:
            Total energy (J)
        """
        kinetic_energy = 0.0
        potential_energy = 0.0

        # Kinetic energy
        for body in self.bodies:
            kinetic_energy += 0.5 * body.mass * np.linalg.norm(body.velocity) ** 2

        # Potential energy
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i < j:  # Avoid double counting
                    r = np.linalg.norm(body2.position - body1.position)
                    if r > 1e-10:  # Avoid division by zero
                        potential_energy -= self.G * body1.mass * body2.mass / r

        return kinetic_energy + potential_energy

    def plot_trajectories(self, save_path: str = None):
        """
        Plot 3D trajectories of all bodies.

        Args:
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.bodies)))

        for i, body in enumerate(self.bodies):
            trajectory = np.array(body.trajectory)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=colors[i],
                label=body.name or f"Body {i+1}",
                linewidth=2,
            )

            # Mark starting position
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color=colors[i],
                s=100,
                marker="o",
                alpha=0.8,
            )

            # Mark current position
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                trajectory[-1, 2],
                color=colors[i],
                s=150,
                marker="*",
                alpha=1.0,
            )

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("N-Body Gravitational Simulation - Trajectories")
        ax.legend()

        # Equal aspect ratio
        max_range = max(
            [
                max(
                    np.array(body.trajectory).max(axis=0)
                    - np.array(body.trajectory).min(axis=0)
                )
                for body in self.bodies
            ]
        )
        ax.set_box_aspect([1, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def animate_simulation(
        self,
        duration: float,
        dt: float,
        update_interval: int = 10,
        trail_length: int = 1000,
        show_vectors: bool = False,
    ):
        """
        Run simulation with real-time 3D animation.

        Args:
            duration: Total simulation time (s)
            dt: Time step (s)
            update_interval: Update plot every N steps
            trail_length: Number of trajectory points to show
            show_vectors: Whether to show velocity vectors
        """
        # Setup the figure and 3D axis
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.bodies)))

        # Initialize plot elements
        lines = []  # Trajectory lines
        points = []  # Current position markers
        velocity_arrows = []  # Velocity vectors

        for i, body in enumerate(self.bodies):
            # Trajectory line
            (line,) = ax.plot(
                [],
                [],
                [],
                color=colors[i],
                linewidth=2,
                alpha=0.7,
                label=body.name or f"Body {i+1}",
            )
            lines.append(line)

            # Current position marker
            point = ax.scatter([], [], [], color=colors[i], s=100, marker="o")
            points.append(point)

            # Velocity vector (if requested)
            if show_vectors:
                arrow = ax.quiver([], [], [], [], [], [], color=colors[i], alpha=0.6)
                velocity_arrows.append(arrow)

        # Text displays
        time_text = ax.text2D(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        energy_text = ax.text2D(
            0.02,
            0.92,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set up axis properties
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_title("Real-Time N-Body Gravitational Simulation")
        ax.legend(loc="upper right")

        # Animation state
        step_count = 0
        total_steps = int(duration / dt)
        initial_energy = self.total_energy()

        def update_plot():
            """Update the 3D plot with current simulation state."""
            # Calculate axis limits based on current positions
            all_positions = np.array([body.position for body in self.bodies])
            if len(self.bodies[0].trajectory) > 1:
                all_trajectories = np.vstack(
                    [np.array(body.trajectory) for body in self.bodies]
                )
                all_positions = np.vstack([all_positions, all_trajectories])

            margin = 0.1
            x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
            y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
            z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2

            ax.set_xlim(
                center_x - max_range / 2 * (1 + margin),
                center_x + max_range / 2 * (1 + margin),
            )
            ax.set_ylim(
                center_y - max_range / 2 * (1 + margin),
                center_y + max_range / 2 * (1 + margin),
            )
            ax.set_zlim(
                center_z - max_range / 2 * (1 + margin),
                center_z + max_range / 2 * (1 + margin),
            )

            # Update trajectories and positions
            for i, body in enumerate(self.bodies):
                trajectory = np.array(body.trajectory)

                # Limit trajectory length for performance
                if len(trajectory) > trail_length:
                    trajectory = trajectory[-trail_length:]

                if len(trajectory) > 1:
                    lines[i].set_data(trajectory[:, 0], trajectory[:, 1])
                    lines[i].set_3d_properties(trajectory[:, 2])

                # Update current position
                points[i]._offsets3d = (
                    [body.position[0]],
                    [body.position[1]],
                    [body.position[2]],
                )

                # Update velocity vectors
                if show_vectors and velocity_arrows:
                    vel_scale = max_range / 1e6  # Scale velocity vectors appropriately
                    velocity_arrows[i].remove()
                    velocity_arrows[i] = ax.quiver(
                        body.position[0],
                        body.position[1],
                        body.position[2],
                        body.velocity[0] * vel_scale,
                        body.velocity[1] * vel_scale,
                        body.velocity[2] * vel_scale,
                        color=colors[i],
                        alpha=0.6,
                        arrow_length_ratio=0.1,
                    )

            # Update information displays
            time_text.set_text(
                f"Time: {self.time:.2e} s\nStep: {step_count}/{total_steps}"
            )

            current_energy = self.total_energy()
            energy_change = (
                abs(current_energy - initial_energy) / abs(initial_energy) * 100
            )
            energy_text.set_text(
                f"Energy: {current_energy:.2e} J\nΔE: {energy_change:.6f}%"
            )

            plt.draw()

        # Animation loop
        plt.ion()  # Turn on interactive mode
        plt.show()

        try:
            # while step_count < total_steps:
            while 1:
                # Perform simulation steps
                for _ in range(update_interval):
                    self.step(dt)
                    step_count += 1

                # if step_count >= total_steps:
                #     break

                # Update visualization
                update_plot()
                plt.pause(0.001)  # Small pause to allow plot update

                # Check if window is still open
                if not plt.get_fignums():
                    break

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            plt.ioff()  # Turn off interactive mode

        print(f"\nSimulation completed!")
        print(f"Final time: {self.time:.2e} s")
        print(f"Final energy: {self.total_energy():.2e} J")

    def animate_simulation_2d(
        self,
        duration: float,
        dt: float,
        update_interval: int = 10,
        trail_length: int = 500,
        projection_plane: str = "xy",
    ):
        """
        Run simulation with real-time 2D animation (faster than 3D).

        Args:
            duration: Total simulation time (s)
            dt: Time step (s)
            update_interval: Update plot every N steps
            trail_length: Number of trajectory points to show
            projection_plane: 'xy', 'xz', or 'yz'
        """
        # Setup the figure and 2D axis
        fig, ax = plt.subplots(figsize=(12, 10))

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.bodies)))

        # Determine coordinate indices based on projection plane
        if projection_plane == "xy":
            coord_indices = (0, 1)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
        elif projection_plane == "xz":
            coord_indices = (0, 2)
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Z Position (m)")
        else:  # 'yz'
            coord_indices = (1, 2)
            ax.set_xlabel("Y Position (m)")
            ax.set_ylabel("Z Position (m)")

        # Initialize plot elements
        lines = []
        points = []

        for i, body in enumerate(self.bodies):
            # Trajectory line
            (line,) = ax.plot(
                [],
                [],
                color=colors[i],
                linewidth=2,
                alpha=0.7,
                label=body.name or f"Body {i+1}",
            )
            lines.append(line)

            # Current position marker
            (point,) = ax.plot([], [], color=colors[i], marker="o", markersize=8)
            points.append(point)

        # Text displays
        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        energy_text = ax.text(
            0.02,
            0.90,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(
            f"Real-Time N-Body Simulation ({projection_plane.upper()} Projection)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Animation state
        step_count = 0
        total_steps = int(duration / dt)
        initial_energy = self.total_energy()

        def update_plot():
            """Update the 2D plot with current simulation state."""
            # Get coordinate data
            all_coords = []
            for body in self.bodies:
                trajectory = np.array(body.trajectory)
                if len(trajectory) > 1:
                    # Limit trajectory length for performance
                    if len(trajectory) > trail_length:
                        trajectory = trajectory[-trail_length:]
                    all_coords.extend(trajectory[:, coord_indices])

            if all_coords:
                all_coords = np.array(all_coords)
                margin = 0.1
                coord1_min, coord1_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                coord2_min, coord2_max = all_coords[:, 1].min(), all_coords[:, 1].max()

                coord1_range = coord1_max - coord1_min
                coord2_range = coord2_max - coord2_min
                max_range = max(coord1_range, coord2_range)

                center1 = (coord1_min + coord1_max) / 2
                center2 = (coord2_min + coord2_max) / 2

                ax.set_xlim(
                    center1 - max_range / 2 * (1 + margin),
                    center1 + max_range / 2 * (1 + margin),
                )
                ax.set_ylim(
                    center2 - max_range / 2 * (1 + margin),
                    center2 + max_range / 2 * (1 + margin),
                )

            # Update trajectories and positions
            for i, body in enumerate(self.bodies):
                trajectory = np.array(body.trajectory)

                # Limit trajectory length for performance
                if len(trajectory) > trail_length:
                    trajectory = trajectory[-trail_length:]

                if len(trajectory) > 1:
                    lines[i].set_data(
                        trajectory[:, coord_indices[0]], trajectory[:, coord_indices[1]]
                    )

                # Update current position
                current_pos = body.position[coord_indices]
                points[i].set_data([current_pos[0]], [current_pos[1]])

            # Update information displays
            time_text.set_text(
                f"Time: {self.time:.2e} s\nStep: {step_count}/{total_steps}"
            )

            current_energy = self.total_energy()
            energy_change = (
                abs(current_energy - initial_energy) / abs(initial_energy) * 100
            )
            energy_text.set_text(
                f"Energy: {current_energy:.2e} J\nΔE: {energy_change:.6f}%"
            )

            plt.draw()

        # Animation loop
        plt.ion()
        plt.show()

        try:
            # while step_count < total_steps:
            while 1:
                # Perform simulation steps
                for _ in range(update_interval):
                    self.step(dt)
                    step_count += 1

                # if step_count >= total_steps:
                #     break

                # Update visualization
                update_plot()
                plt.pause(0.001)

                # Check if window is still open
                if not plt.get_fignums():
                    break

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            plt.ioff()

        print(f"\nSimulation completed!")
        print(f"Final time: {self.time:.2e} s")
        print(f"Final energy: {self.total_energy():.2e} J")


def create_solar_system_example():
    """
    Create a simplified solar system example with Sun, Earth, and Moon.
    """
    # Units: kg, m, m/s
    # Simplified masses and distances for demonstration

    # Sun
    sun = Body(
        mass=1.989e30,  # kg
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        name="Sun",
    )

    # Earth
    earth = Body(
        mass=5.972e24,  # kg
        position=[1.496e11, 0, 0],  # 1 AU from Sun
        velocity=[0, 29780, 0],  # Orbital velocity
        name="Earth",
    )

    # Moon (relative to Earth)
    moon = Body(
        mass=7.342e22,  # kg
        position=[1.496e11 + 3.844e8, 0, 0],  # Earth position + lunar distance
        velocity=[0, 29780 + 1022, 0],  # Earth velocity + lunar orbital velocity
        name="Moon",
    )

    return [sun, earth, moon]


def create_binary_star_example():
    """
    Create a binary star system example.
    """
    # Two stars of equal mass orbiting their common center
    star1 = Body(
        mass=2e30,  # kg
        position=[-1e11, 0, 0],  # m
        velocity=[0, -15000, 0],  # m/s
        name="Star 1",
    )

    star2 = Body(
        mass=2e30,  # kg
        position=[1e11, 0, 0],  # m
        velocity=[0, 15000, 0],  # m/s
        name="Star 2",
    )

    return [star1, star2]


# Example usage
if __name__ == "__main__":
    # Choose your simulation
    print("N-Body Gravitational Simulation")
    print("1. Solar System (Sun-Earth-Moon)")
    print("2. Binary Star System")
    print("3. Custom System")

    choice = input("Choose simulation (1, 2, or 3): ")

    if choice == "1":
        bodies = create_solar_system_example()
        simulation_time = 365.25 * 24 * 3600  # 1 year in seconds
        dt = 3600 * 6  # 6 hour time step for faster animation
    elif choice == "2":
        bodies = create_binary_star_example()
        simulation_time = 365.25 * 24 * 3600  # 1 year in seconds
        dt = 3600 * 12  # 12 hour time step
    else:
        # Custom system example - three body problem
        bodies = [
            Body(
                mass=2e30,
                position=[-1e11, 0, 0],
                velocity=[0, -10000, 0],
                name="Body 1",
            ),
            Body(
                mass=2e30, position=[1e11, 0, 0], velocity=[0, 10000, 0], name="Body 2"
            ),
            Body(
                mass=1e30, position=[0, 1e11, 0], velocity=[15000, 0, 0], name="Body 3"
            ),
        ]
        simulation_time = 365.25 * 24 * 3600
        dt = 3600 * 12

    # Create simulation
    sim = NBodySimulation(bodies)

    print(f"Initial total energy: {sim.total_energy():.2e} J")

    # Choose visualization type
    print("\nVisualization Options:")
    print("1. Real-time 3D animation")
    print("2. Real-time 2D animation (XY plane)")
    print("3. Real-time 2D animation (XZ plane)")
    print("4. Real-time 2D animation (YZ plane)")
    print("5. Static plot after simulation")

    vis_choice = input("Choose visualization (1-5): ")

    if vis_choice == "1":
        # 3D real-time animation
        print("\nStarting 3D real-time simulation...")
        print("Press Ctrl+C to stop early, or close the window")
        sim.animate_simulation(
            simulation_time, dt, update_interval=5, trail_length=2000
        )

    elif vis_choice in ["2", "3", "4"]:
        # 2D real-time animation
        plane_map = {"2": "xy", "3": "xz", "4": "yz"}
        plane = plane_map[vis_choice]
        print(f"\nStarting 2D real-time simulation ({plane.upper()} plane)...")
        print("Press Ctrl+C to stop early, or close the window")
        sim.animate_simulation_2d(
            simulation_time,
            dt,
            update_interval=5,
            trail_length=1000,
            projection_plane=plane,
        )

    else:
        # Static simulation
        print("\nRunning simulation without real-time visualization...")
        sim.simulate(simulation_time, dt)
        print(f"Final total energy: {sim.total_energy():.2e} J")
        sim.plot_trajectories()
